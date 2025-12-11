from thulium.data.language_profiles import SUPPORTED_LANGUAGES

class MultiLanguagePipeline:
    """
    Pipeline that supports routing to different language models based on
    page-level or region-level classification.
    """
    def __init__(self, languages: list):
        self.languages = languages
    
    def detect_language(self, image) -> str:
        # Stub language ID
        return "en"
