import ast
import importlib.metadata
import inspect
import io
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from deployment import runtime
from deployment.gradio_app import create_app


def complete_dependency_probes():
    return {
        "version_lookup": lambda name: runtime.PINNED_VERSIONS.get(name),
        "module_lookup": lambda name: True,
        "executable_lookup": lambda name: f"/usr/bin/{name}",
    }


def settings_for(root: Path, *, hashes=None, max_upload_bytes=1024):
    root.mkdir(parents=True, exist_ok=True)
    names = (
        ("audio_base", "audio/BEATs_iter3.pt"),
        ("audio_trained", "audio/beats_lora_r8_v1/best.pt"),
        ("audio_normalization", "audio/beats_iter3/norm_stats.json"),
        ("vision_trained", "vision/resnet18_v1/best.pt"),
    )
    artifacts = tuple(runtime.ArtifactSetting(name, root / rel, (hashes or {}).get(name))
                      for name, rel in names)
    return runtime.DeploymentConfig(root, artifacts,
                                    (("E3", root / "fusion/E3"), ("A2", root / "fusion/A2")),
                                    max_upload_bytes)


def valid_media(suffix=".mp4"):
    profiles = {
        ".mp4": (("mov", "mp4"), "h264", "aac", (("major_brand", "mp42"),)),
        ".mov": (("mov", "mp4"), "h264", "aac", (("major_brand", "qt  "),)),
        ".mkv": (("matroska", "webm"), "h264", "aac", ()),
        ".webm": (("matroska", "webm"), "vp9", "opus", ()),
    }
    formats, video_codec, audio_codec, tags = profiles[suffix]
    return runtime.MediaInfo(
        formats, tags,
        (runtime.MediaStream("video", video_codec, 10.0),
         runtime.MediaStream("audio", audio_codec, 10.0)),
    )


def valid_auth_environment():
    return {
        "AVAL_ACCESS_MODE": "gradio_auth",
        "AVAL_GRADIO_USERNAME": "space-user",
        "AVAL_GRADIO_PASSWORD": "synthetic-test-password",
    }


class DeploymentConfigurationTests(unittest.TestCase):
    def test_missing_artifacts_and_raw_gates_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            report = runtime.startup_readiness(settings_for(Path(temp)), **complete_dependency_probes())
        self.assertEqual(report.status, "MODEL_NOT_READY")
        self.assertFalse(report.inference_enabled)
        codes = {(issue.category, issue.code) for issue in report.issues}
        self.assertIn(("artifact:audio_base", "MISSING_ARTIFACT"), codes)
        self.assertIn(("artifact:vision_trained", "MISSING_ARTIFACT"), codes)
        self.assertIn(("raw_media_decoder", "NOT_IMPLEMENTED"), codes)
        self.assertIn(("raw_to_train_export_parity", "NOT_VERIFIED"), codes)

    def test_configurable_paths_and_hashes_are_root_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models"
            config = runtime.DeploymentConfig.from_environment({
                "AVAL_MODEL_ROOT": str(root),
                "AVAL_AUDIO_BASE_PATH": str(root / "custom" / "base.pt"),
                "AVAL_AUDIO_BASE_SHA256": "a" * 64,
                "AVAL_MAX_UPLOAD_BYTES": "1024",
            })
            self.assertEqual(config.artifacts[0].path, (root / "custom" / "base.pt").resolve())
            self.assertEqual(config.artifacts[0].sha256, "a" * 64)
            self.assertEqual(config.max_upload_bytes, 1024)
            with self.assertRaisesRegex(runtime.DeploymentError, "ARTIFACT_PATH_OUTSIDE_MODEL_ROOT"):
                runtime.DeploymentConfig.from_environment({
                    "AVAL_MODEL_ROOT": str(root),
                    "AVAL_AUDIO_BASE_PATH": str(Path(temp).parent / "escape.pt"),
                })

    def test_bad_upload_limit_rejected(self):
        with self.assertRaisesRegex(runtime.DeploymentError, "INVALID_UPLOAD_LIMIT"):
            runtime.DeploymentConfig.from_environment({"AVAL_MAX_UPLOAD_BYTES": "0"})

    def test_wrong_hash_rejected_before_any_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / "audio/BEATs_iter3.pt"
            file.parent.mkdir(parents=True)
            file.write_bytes(b"synthetic checkpoint bytes")
            config = settings_for(root, hashes={
                "audio_base": "0" * 64,
                "audio_trained": "1" * 64,
                "audio_normalization": "2" * 64,
                "vision_trained": "3" * 64,
            })
            with mock.patch("torch.load") as torch_load:
                with self.assertRaisesRegex(runtime.DeploymentError, "HASH_MISMATCH"):
                    runtime.load_verified_encoder_pair(config)
            torch_load.assert_not_called()

    def test_runtime_and_decoder_dependencies_reported_by_category(self):
        with tempfile.TemporaryDirectory() as temp:
            missing_version = lambda name: None if name == "torchaudio" else runtime.PINNED_VERSIONS[name]
            missing_executable = lambda name: None if name == "ffprobe" else f"/usr/bin/{name}"
            report = runtime.startup_readiness(
                settings_for(Path(temp)), version_lookup=missing_version,
                module_lookup=lambda _: True, executable_lookup=missing_executable)
        pairs = {(item.category, item.code) for item in report.issues}
        self.assertIn(("dependency:torchaudio", "DEPENDENCY_VERSION_MISMATCH"), pairs)
        self.assertIn(("decoder:ffprobe", "DECODER_DEPENDENCY_MISSING"), pairs)

    def test_pillow_is_a_required_versioned_runtime_dependency(self):
        self.assertEqual(runtime.PINNED_VERSIONS["pillow"], "12.0.0")
        with tempfile.TemporaryDirectory() as temp:
            wrong = lambda name: "11.0.0" if name == "pillow" else runtime.PINNED_VERSIONS[name]
            report = runtime.startup_readiness(
                settings_for(Path(temp)), version_lookup=wrong,
                module_lookup=lambda _: True, executable_lookup=lambda name: f"/bin/{name}")
        self.assertIn(("dependency:pillow", "DEPENDENCY_VERSION_MISMATCH"),
                      {(item.category, item.code) for item in report.issues})

    def test_access_configuration_requires_supported_mode_and_credentials(self):
        cases = (
            {},
            {"AVAL_ACCESS_MODE": "private_space"},
            {"AVAL_ACCESS_MODE": "gradio_auth", "AVAL_GRADIO_USERNAME": "user"},
            {**valid_auth_environment(), "AVAL_ACCESS_MODE": "unknown"},
        )
        for values in cases:
            config = runtime.DeploymentConfig.from_environment(values)
            self.assertFalse(config.authenticated_uploads_enabled)
            self.assertIsNone(config.gradio_auth)
        good = runtime.DeploymentConfig.from_environment(valid_auth_environment())
        self.assertTrue(good.authenticated_uploads_enabled)
        self.assertEqual(good.gradio_auth, ("space-user", "synthetic-test-password"))
        self.assertNotIn("synthetic-test-password", repr(good))

    def test_frozen_fusion_candidate_bindings_match_phase6(self):
        self.assertEqual(set(runtime.CANDIDATES), {"E3", "A2"})
        self.assertEqual(runtime.CANDIDATES["E3"].threshold, 0.009579099009671126)
        self.assertEqual(runtime.CANDIDATES["A2"].threshold, 0.0026640388978268056)
        with self.assertRaisesRegex(runtime.DeploymentError, "UNKNOWN_FROZEN_CANDIDATE"):
            runtime.load_frozen_fusion(Path("/unused"), "RANDOM")


class UploadAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = settings_for(Path(self.temp.name), max_upload_bytes=32)
        self.readiness = runtime.ReadinessReport(
            "MODEL_NOT_READY",
            (runtime.ReadinessIssue("artifact:audio_base", "MISSING_ARTIFACT"),),
            False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_upload_and_oversize_are_rejected_before_probe(self):
        def probe(_):
            self.fail("probe must not run")
        self.assertEqual(runtime.validate_upload(None, config=self.config,
                         filename=None, readiness=self.readiness, probe=probe)["error_code"], "UPLOAD_REQUIRED")
        self.assertEqual(runtime.validate_upload(b"x" * 33, config=self.config,
                         filename="clip.mp4", readiness=self.readiness, probe=probe)["error_code"], "UPLOAD_TOO_LARGE")

    def test_unsupported_container_and_decoder_error_are_explicit(self):
        response = runtime.validate_upload(
            b"not-a-container", filename="clip.mp4", config=self.config, readiness=self.readiness,
            probe=lambda _: (_ for _ in ()).throw(runtime.DeploymentError("UNSUPPORTED_UPLOAD")))
        self.assertEqual(response["status"], "REJECTED")
        self.assertEqual(response["error_code"], "UNSUPPORTED_UPLOAD")
        missing = runtime.validate_upload(
            b"container", filename="clip.mp4", config=self.config, readiness=self.readiness,
            probe=lambda _: (_ for _ in ()).throw(runtime.DeploymentError("DECODER_DEPENDENCY_MISSING")))
        self.assertEqual(missing["status"], "MODEL_NOT_READY")
        self.assertEqual(missing["error_code"], "DECODER_DEPENDENCY_MISSING")

    def test_missing_modality_and_invalid_duration_are_rejected(self):
        no_audio = runtime.validate_upload(
            b"container", filename="clip.mp4", config=self.config, readiness=self.readiness,
            probe=lambda _: runtime.MediaInfo(("mp4",), (("major_brand", "mp42"),),
                (runtime.MediaStream("video", "h264", 10.0),)))
        self.assertEqual(no_audio["error_code"], "MISSING_OR_AMBIGUOUS_STREAMS")
        bad_duration = runtime.validate_upload(
            b"container", filename="clip.mp4", config=self.config, readiness=self.readiness,
            probe=lambda _: runtime.MediaInfo(("mp4",), (("major_brand", "mp42"),),
                (runtime.MediaStream("video", "h264", 9.0),
                 runtime.MediaStream("audio", "aac", 10.0))))
        self.assertEqual(bad_duration["error_code"], "INVALID_DURATION")
        mismatch = runtime.validate_upload(
            b"container", filename="clip.mp4", config=self.config, readiness=self.readiness,
            probe=lambda _: runtime.MediaInfo(("mp4",), (("major_brand", "mp42"),),
                (runtime.MediaStream("video", "h264", 9.96),
                 runtime.MediaStream("audio", "aac", 10.04))))
        self.assertEqual(mismatch["error_code"], "MODALITY_DURATION_MISMATCH")

    def test_extension_container_and_codec_are_checked_server_side(self):
        wrong_extension = runtime.validate_upload(
            b"synthetic", filename="clip.webm", config=self.config, readiness=self.readiness,
            probe=lambda _: valid_media(".mp4"))
        self.assertEqual(wrong_extension["error_code"], "UNSUPPORTED_CONTAINER")
        mov_extension_with_mp4_brand = runtime.validate_upload(
            b"synthetic", filename="clip.mov", config=self.config, readiness=self.readiness,
            probe=lambda _: valid_media(".mp4"))
        self.assertEqual(mov_extension_with_mp4_brand["error_code"], "UNSUPPORTED_CONTAINER")
        unsupported_codec = runtime.validate_upload(
            b"synthetic", filename="clip.mp4", config=self.config, readiness=self.readiness,
            probe=lambda _: runtime.MediaInfo(("mp4",), (("major_brand", "mp42"),),
                (runtime.MediaStream("video", "hevc", 10.0),
                 runtime.MediaStream("audio", "aac", 10.0))))
        self.assertEqual(unsupported_codec["error_code"], "UNSUPPORTED_CODEC")
        unsupported_suffix = runtime.validate_upload(
            b"synthetic", filename="clip.avi", config=self.config, readiness=self.readiness,
            probe=lambda _: self.fail("unknown extensions must reject before ffprobe"))
        self.assertEqual(unsupported_suffix["error_code"], "UNSUPPORTED_EXTENSION")

    def test_multiple_streams_and_malformed_metadata_reject(self):
        extra = runtime.MediaInfo(("mp4",), (("major_brand", "mp42"),), (*valid_media().streams,
            runtime.MediaStream("audio", "aac", 10.0)))
        result = runtime.validate_upload(b"synthetic", filename="clip.mp4", config=self.config,
                                         readiness=self.readiness, probe=lambda _: extra)
        self.assertEqual(result["error_code"], "MISSING_OR_AMBIGUOUS_STREAMS")
        malformed = runtime.validate_upload(b"synthetic", filename="clip.mp4", config=self.config,
            readiness=self.readiness, probe=lambda _: {"streams": []})
        self.assertEqual(malformed["error_code"], "MALFORMED_MEDIA_METADATA")

    def test_private_temporary_upload_is_removed_after_valid_probe(self):
        paths = []
        def probe(path):
            paths.append(path)
            self.assertTrue(path.is_file())
            return valid_media()
        result = runtime.validate_upload(b"synthetic container", filename="clip.mp4", config=self.config,
                                         readiness=self.readiness, probe=probe)
        self.assertEqual(result["status"], "MODEL_NOT_READY")
        self.assertIsNone(result["predictions"])
        self.assertEqual(result["error_code"], "RAW_INFERENCE_GATE_CLOSED")
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists())

    def test_private_temporary_upload_is_removed_after_probe_failure(self):
        paths = []
        def probe(path):
            paths.append(path)
            raise RuntimeError("secret filename and media content")
        result = runtime.validate_upload(b"synthetic container", filename="clip.mp4", config=self.config,
                                         readiness=self.readiness, probe=probe)
        self.assertEqual(result["error_code"], "UPLOAD_VALIDATION_FAILED")
        self.assertFalse(paths[0].exists())

    def test_logs_never_include_uploaded_bytes_or_secret_values(self):
        secret = b"hf_fake_secret_value"
        with self.assertLogs("avalidate.deployment", level="INFO") as captured:
            result = runtime.validate_upload(
                secret, filename="clip.mp4", config=self.config, readiness=self.readiness,
                probe=lambda _: (_ for _ in ()).throw(runtime.DeploymentError("UNSUPPORTED_UPLOAD")))
        self.assertEqual(result["status"], "REJECTED")
        self.assertNotIn(secret.decode(), "\n".join(captured.output))
        self.assertNotIn("fake_secret", "\n".join(captured.output))

    def test_phase6_gate_rejects_nonpipeline_and_synthetic_modes(self):
        with self.assertRaisesRegex(runtime.DeploymentError, "PHASE6_PIPELINE_NOT_VERIFIED"):
            runtime.run_phase6_contract({"status": "TRAIN_EXPORT_PARITY_MATCHED"}, object())
        from fusion.end_to_end import MediaFusionPipeline
        pipeline = object.__new__(MediaFusionPipeline)
        object.__setattr__(pipeline, "synthetic", True)
        with self.assertRaisesRegex(runtime.DeploymentError, "PHASE6_PIPELINE_NOT_VERIFIED"):
            runtime.run_phase6_contract(pipeline, object())

    def test_deployment_has_no_annotation_or_research_app_fallback(self):
        source = Path("deployment/gradio_app.py").read_text(encoding="utf-8")
        self.assertNotIn("import app", source)
        self.assertNotIn("Annotations.txt", source)
        self.assertIn("predictions are disabled", source)

    def test_gradio_missing_dependency_has_sanitized_startup_error(self):
        with mock.patch.dict("sys.modules", {"gradio": None}):
            with self.assertRaisesRegex(RuntimeError, "GRADIO_DEPENDENCY_MISSING"):
                create_app()

    def test_gradio_ui_uses_filepath_upload_private_endpoint_and_authenticated_launch(self):
        from deployment import gradio_app
        components = []
        class Blocks:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def queue(self, **kwargs): self.queue_options = kwargs
            def launch(self, **kwargs): self.launch_options = kwargs
            def close(self): self.closed = True
        class Component:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                components.append(self)
            def click(self, **kwargs): self.click_options = kwargs
            def change(self, **kwargs): self.change_options = kwargs
            def __enter__(self): return self
            def __exit__(self, *_): return False
        fake = type("FakeGradio", (), {
            "Request": object,
            "HTML": Component,
            "Tabs": Component,
            "Tab": Component,
            "Accordion": Component,
            "Blocks": Blocks,
            "Markdown": staticmethod(lambda *_args, **_kwargs: None),
            "File": Component,
            "Textbox": Component,
            "Button": Component,
            "JSON": Component,
        })
        with mock.patch.dict(sys.modules, {"gradio": fake}), mock.patch.dict(
            "os.environ", valid_auth_environment(), clear=True):
            demo, config = create_app()
        with mock.patch.object(gradio_app, "create_app", return_value=(demo, config)):
            gradio_app.main()
        self.assertEqual(demo.queue_options["default_concurrency_limit"], 1)
        self.assertFalse(demo.queue_options["api_open"])
        self.assertEqual(config.max_upload_bytes, runtime.MAX_UPLOAD_BYTES_HARD_LIMIT)
        upload = next(component for component in components if "file_types" in component.kwargs)
        self.assertEqual(upload.kwargs["type"], "filepath")
        button = next(component for component in components if hasattr(component, "click_options"))
        self.assertEqual(button.click_options["api_visibility"], "private")
        self.assertEqual(button.click_options["concurrency_limit"], 1)
        self.assertFalse(demo.kwargs["analytics_enabled"])
        self.assertEqual(demo.launch_options["auth"], config.gradio_auth)
        self.assertTrue(demo.launch_options["prevent_thread_lock"])
        self.assertTrue(demo.closed)

    def test_gradio_calls_match_the_pinned_blocks_api_signatures(self):
        """Guard the real application call against the Gradio 6.30.0 API."""
        expected_version = "6.30.0"
        self.assertEqual(runtime.PINNED_VERSIONS["gradio"], expected_version)

        source = Path("deployment/gradio_app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        blocks_keywords = set()
        launch_keywords = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "Blocks":
                blocks_keywords.update(keyword.arg for keyword in node.keywords if keyword.arg)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "launch":
                launch_keywords.update(keyword.arg for keyword in node.keywords if keyword.arg)

        self.assertIn("analytics_enabled", blocks_keywords)
        self.assertNotIn("analytics_enabled", launch_keywords)

        # Parameters from Gradio's tagged 6.30.0 Blocks.launch signature.
        pinned_launch_parameters = {
            "inline", "inbrowser", "share", "debug", "max_threads", "auth",
            "auth_message", "prevent_thread_lock", "show_error", "server_name",
            "server_port", "height", "width", "favicon_path", "ssl_keyfile",
            "ssl_certfile", "ssl_keyfile_password", "ssl_verify", "quiet",
            "footer_links", "run_history", "allowed_paths", "blocked_paths",
            "root_path", "app_kwargs", "state_session_capacity", "share_server_address",
            "share_server_protocol", "share_server_tls_certificate", "auth_dependency",
            "max_file_size", "enable_monitoring", "strict_cors", "node_server_name",
            "node_port", "ssr_mode", "pwa", "mcp_server", "num_workers", "_app",
        }
        self.assertLessEqual(launch_keywords, pinned_launch_parameters)

        try:
            installed_version = importlib.metadata.version("gradio")
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        if installed_version is not None:
            import gradio

            self.assertEqual(installed_version, expected_version)
            blocks_signature = inspect.signature(gradio.Blocks)
            launch_signature = inspect.signature(gradio.Blocks.launch)
            self.assertIn("analytics_enabled", blocks_signature.parameters)
            self.assertTrue(launch_keywords <= set(launch_signature.parameters))

    def test_app_refuses_to_start_without_supported_auth_configuration(self):
        fake = type("FakeGradio", (), {})
        with mock.patch.dict(sys.modules, {"gradio": fake}), mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AUTHENTICATION_CONFIGURATION_REQUIRED"):
                create_app()

    def test_direct_callback_disables_upload_without_authentication(self):
        from deployment.gradio_app import DeploymentUI
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clip.mp4"
            path.write_bytes(b"synthetic")
            config = settings_for(Path(temp) / "models")
            ui = DeploymentUI(config, self.readiness)
            result = ui.submit(path)
            self.assertEqual(result["error_code"], "AUTHENTICATION_CONFIGURATION_REQUIRED")
            self.assertTrue(path.exists())

    def test_authenticated_file_callback_cleans_managed_upload_if_admission_raises(self):
        from deployment.gradio_app import DeploymentUI
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "clip.mp4"
            path.write_bytes(b"synthetic upload")
            config = runtime.DeploymentConfig.from_environment({
                **valid_auth_environment(), "AVAL_MODEL_ROOT": str(root / "models"),
            })
            ui = DeploymentUI(config, self.readiness)
            with mock.patch.dict("os.environ", {"GRADIO_TEMP_DIR": str(root)}), \
                 mock.patch("deployment.gradio_app.validate_upload", side_effect=RuntimeError("private")):
                with self.assertRaisesRegex(RuntimeError, "private"):
                    ui.submit(path)
            self.assertFalse(path.exists())

    def test_authenticated_file_callback_checks_upload_root_and_cleans_copy(self):
        from deployment.gradio_app import DeploymentUI
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "clip.mp4"
            path.write_bytes(b"synthetic upload")
            config = runtime.DeploymentConfig.from_environment({
                "AVAL_MODEL_ROOT": str(root / "models"), **valid_auth_environment(),
                "AVAL_MAX_UPLOAD_BYTES": "64",
            })
            ui = DeploymentUI(config, self.readiness)
            with mock.patch.dict("os.environ", {"GRADIO_TEMP_DIR": str(root)}), \
                 mock.patch("deployment.gradio_app.validate_upload",
                            return_value={"status": "MODEL_NOT_READY", "predictions": None}) as admitted:
                result = ui.submit(path)
            self.assertEqual(result["status"], "MODEL_NOT_READY")
            admitted.assert_called_once()
            self.assertEqual(admitted.call_args.kwargs["filename"], "clip.mp4")
            self.assertEqual(admitted.call_args.args[0], b"synthetic upload")
            self.assertFalse(path.exists())

    def test_authenticated_file_callback_rejects_path_outside_gradio_root(self):
        from deployment.gradio_app import DeploymentUI
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside.mp4"
            outside.write_bytes(b"synthetic")
            config = runtime.DeploymentConfig.from_environment({**valid_auth_environment()})
            ui = DeploymentUI(config, self.readiness)
            with mock.patch.dict("os.environ", {"GRADIO_TEMP_DIR": str(root / "uploads")}):
                result = ui.submit(outside)
            self.assertEqual(result["error_code"], "UNSUPPORTED_UPLOAD")
            self.assertTrue(outside.exists())

    def test_credential_free_startup_and_bad_credentials_fail_safely(self):
        fake = type("FakeGradio", (), {})
        config = runtime.DeploymentConfig.from_environment({
            "AVAL_ACCESS_MODE": "gradio_auth",
            "AVAL_GRADIO_USERNAME": "user",
            "AVAL_GRADIO_PASSWORD": "",
        })
        self.assertFalse(config.authenticated_uploads_enabled)
        with mock.patch.dict(sys.modules, {"gradio": fake}), mock.patch.dict(
            "os.environ", {"AVAL_ACCESS_MODE": "gradio_auth", "AVAL_GRADIO_USERNAME": "user",
                           "AVAL_GRADIO_PASSWORD": ""}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AUTHENTICATION_CONFIGURATION_REQUIRED") as caught:
                create_app()
        self.assertNotIn("user", str(caught.exception))

    def test_probe_output_parser_requires_container_and_stream_metadata(self):
        payload = (b'{"format":{"format_name":"mov,mp4","tags":{"major_brand":"mp42"}},"streams":['
                   b'{"codec_type":"video","codec_name":"h264","duration":"10"},'
                   b'{"codec_type":"audio","codec_name":"aac","duration":"10"}]}')
        with mock.patch.object(runtime, "_run_bounded_process", return_value=(0, payload, b"")), \
             mock.patch.object(runtime.shutil, "which", return_value="/bin/ffprobe"):
            parsed = runtime.probe_media(Path("unused"))
        self.assertEqual(parsed, valid_media())
        missing_metadata = b'{"format":{"format_name":"mp4"},"streams":[]}'
        with mock.patch.object(runtime, "_run_bounded_process", return_value=(0, missing_metadata, b"")), \
             mock.patch.object(runtime.shutil, "which", return_value="/bin/ffprobe"):
            with self.assertRaisesRegex(runtime.DeploymentError, "UNSUPPORTED_UPLOAD"):
                runtime.probe_media(Path("unused"))

    def test_probe_timeout_and_output_caps_are_explicit(self):
        with mock.patch.object(runtime, "_run_bounded_process",
                               side_effect=runtime.DeploymentError("PROBE_TIMEOUT")), \
             mock.patch.object(runtime.shutil, "which", return_value="/bin/ffprobe"):
            with self.assertRaisesRegex(runtime.DeploymentError, "PROBE_TIMEOUT"):
                runtime.probe_media(Path("unused"))
        with mock.patch.object(runtime, "_run_bounded_process",
                               side_effect=runtime.DeploymentError("PROBE_OUTPUT_TOO_LARGE")), \
             mock.patch.object(runtime.shutil, "which", return_value="/bin/ffprobe"):
            with self.assertRaisesRegex(runtime.DeploymentError, "PROBE_OUTPUT_TOO_LARGE"):
                runtime.probe_media(Path("unused"))

    def test_bounded_probe_runner_kills_on_timeout_and_caps_combined_output(self):
        class FakeProcess:
            def __init__(self, stdout=b"", stderr=b"", timeout=False):
                self.stdout = io.BytesIO(stdout)
                self.stderr = io.BytesIO(stderr)
                self.killed = False
                self.timeout = timeout
            def wait(self, timeout=None):
                if timeout is not None and self.timeout:
                    raise runtime.subprocess.TimeoutExpired("ffprobe", timeout)
                return -9 if self.killed else 0
            def kill(self):
                self.killed = True

        process = FakeProcess(timeout=True)
        with mock.patch.object(runtime.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(runtime.DeploymentError, "PROBE_TIMEOUT"):
                runtime._run_bounded_process(["ffprobe"])
        self.assertTrue(process.killed)

        process = FakeProcess(stdout=b"x" * (runtime.MAX_PROBE_OUTPUT_BYTES + 1))
        with mock.patch.object(runtime.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(runtime.DeploymentError, "PROBE_OUTPUT_TOO_LARGE"):
                runtime._run_bounded_process(["ffprobe"])
        self.assertTrue(process.killed)


if __name__ == "__main__":
    unittest.main()
