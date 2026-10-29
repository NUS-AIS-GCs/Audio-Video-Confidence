// Loaded only by the explicit local research section; never the ordinary live UI.
const AVLiveResearch = {
  render: function (element, result) {
    element.textContent = "RESEARCH ONLY / NOT APPROVED\n" + result.classification +
      "\nActual raw E3 score: " + result.score + "\nFrozen threshold: " + result.threshold +
      "\nAt/above threshold: Original-like. Below: Audio-substituted-like." +
      "\nUncalibrated correspondence; this does not prove physical sound causation.\n" + JSON.stringify(result,null,2);
  }
};
if (typeof module !== "undefined" && module.exports) module.exports = AVLiveResearch;
