import unittest
from pathlib import Path

from deployment import runtime


class BackendResponseContractTests(unittest.TestCase):
    def test_supported_upload_keeps_current_response_fail_closed(self):
        config = runtime.DeploymentConfig(Path("/models"), (), ())
        readiness = runtime.ReadinessReport(
            "MODEL_NOT_READY",
            (
                runtime.ReadinessIssue("raw_media_decoder", "NOT_IMPLEMENTED"),
                runtime.ReadinessIssue("raw_to_train_export_parity", "NOT_VERIFIED"),
            ),
            False,
        )
        media = runtime.MediaInfo(
            ("mp4",),
            (("major_brand", "mp42"),),
            (
                runtime.MediaStream("video", "h264", 10.0),
                runtime.MediaStream("audio", "aac", 10.0),
            ),
        )

        result = runtime.validate_upload(
            b"synthetic-container-fixture",
            filename="fixture.mp4",
            config=config,
            readiness=readiness,
            probe=lambda _path: media,
        )

        self.assertEqual(result["status"], "MODEL_NOT_READY")
        self.assertIsNone(result["predictions"])
        self.assertEqual(result["error_code"], "RAW_INFERENCE_GATE_CLOSED")
        self.assertEqual(
            result["issues"],
            [
                {"category": "raw_media_decoder", "code": "NOT_IMPLEMENTED"},
                {"category": "raw_to_train_export_parity", "code": "NOT_VERIFIED"},
            ],
        )
        self.assertNotIn("score", result)
        self.assertNotIn("logit", result)


if __name__ == "__main__":
    unittest.main()
