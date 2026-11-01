"""
voice/ — Module multimodal pour l'agent conversationnel TechShop

4 modes d'interaction disponibles :
  - voice_to_voice  : micro → Whisper STT → agent → pyttsx3 TTS → haut-parleur
  - text_to_voice   : texte → agent → pyttsx3 TTS → haut-parleur
  - voice_to_text   : micro → Whisper STT → agent → texte affiché
  - text_to_text    : texte → agent → texte affiché  (mode standard S1-S4)

Tous les modèles tournent en LOCAL — aucune clé API nécessaire.
"""
