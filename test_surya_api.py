try:
    from surya.inference import SuryaInferenceManager
    print("Inference Manager OK")
except Exception as e:
    print("Inference Manager Error:", e)

try:
    from surya.layout import LayoutPredictor
    print("LayoutPredictor OK")
except Exception as e:
    print("LayoutPredictor Error:", e)
