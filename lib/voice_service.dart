import 'dart:async';
import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';

class VoiceService {
  // Singleton pattern
  static final VoiceService _instance = VoiceService._internal();
  factory VoiceService() => _instance;
  VoiceService._internal();
  

  final stt.SpeechToText _speech = stt.SpeechToText();
  final FlutterTts _flutterTts = FlutterTts();
  final String libreTranslateUrl = "https://libretranslate.de"; // or "https://translate.argosopentech.com"
  
  bool _isInitialized = false;
  bool _isListening = false;
  bool _isSpeaking = false; // Track if text is being spoken
  
  // Initialize speech services
  Future<bool> initialize() async {
    if (_isInitialized) return true;
    
    // Initialize speech to text
    bool speechAvailable = await _speech.initialize(
      onStatus: (status) {
        if (status == 'done') {
          _isListening = false;
        }
      },
      onError: (error) {
        print('Speech to text error: $error');
        _isListening = false;
      },
    );
    
    // Initialize text to speech
    await _flutterTts.setLanguage("hi-IN");
    await _flutterTts.setSpeechRate(0.5);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.0);
    
    _isInitialized = speechAvailable;
    return _isInitialized;
  }
  
  // Check if speech recognition is available
  bool get isAvailable => _isInitialized;
  
  // Check if currently listening
  bool get isListening => _isListening;
  
  // Start listening for speech
  Future<void> startListening(Function(String) onResult) async {
    if (!_isInitialized) {
      bool initialized = await initialize();
      if (!initialized) return;
    }
    
    if (!_isListening) {
      _isListening = await _speech.listen(
        onResult: (result) {
          if (result.finalResult) {
            onResult(result.recognizedWords);
          }
        },
        localeId: "en_IN",
      );
    }
  }
  
  // Stop listening
  Future<void> stopListening() async {
    if (_isListening) {
      await _speech.stop();
      _isListening = false;
    }
  }
  
  // Speak text
  Future<void> speak(String text) async {
    if (!_isInitialized) {
      await initialize();
    }
    
    if (!_isSpeaking) {
      _isSpeaking = true;
      await _flutterTts.speak(text);
    }
  }
  
  // Stop speaking
  Future<void> stopSpeaking() async {
    if (_isSpeaking) {
      await _flutterTts.stop();
      _isSpeaking = false;
    }
  }
  
  // Dispose resources
  Future<void> dispose() async {
    await _flutterTts.stop();
    _speech.cancel();
  }
  
  // UI Widget for voice input button
  Widget buildVoiceInputButton(BuildContext context, TextEditingController textController) {
    return IconButton(
      icon: AnimatedSwitcher(
        duration: const Duration(milliseconds: 200),
        transitionBuilder: (Widget child, Animation<double> animation) {
          return ScaleTransition(scale: animation, child: child);
        },
        child: Icon(
          _isListening ? Icons.mic : Icons.mic_none,
          key: ValueKey<bool>(_isListening),
          color: _isListening ? Colors.red : null,
        ),
      ),
      onPressed: () async {
        if (_isListening) {
          await stopListening();
        } else {
          bool available = await initialize();
          if (available) {
            await startListening((text) {
              if (text.isNotEmpty) {
                textController.text = text;
              }
            });
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Speech recognition not available')),
            );
          }
        }
      },
      tooltip: _isListening ? 'Stop listening' : 'Start voice input',
    );
  }
  
  // UI Widget to play/stop response
  Widget buildTtsButton(BuildContext context, String text) {
    return IconButton(
      icon: Icon(
        _isSpeaking ? Icons.volume_off : Icons.volume_up,
      ),
      onPressed: () async {
        if (_isSpeaking) {
          await stopSpeaking();
        } else {
          await speak(text);
        }
      },
      tooltip: _isSpeaking ? 'Stop narration' : 'Read response aloud',
    );
  }
}