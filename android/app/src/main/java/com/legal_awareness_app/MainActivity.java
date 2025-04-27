package com.yourcompany.legal_awareness_app;

import androidx.annotation.NonNull;
import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugins.GeneratedPluginRegistrant;

public class MainActivity extends FlutterActivity {
    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        // Register all plugins
        GeneratedPluginRegistrant.registerWith(flutterEngine);
        
        // Add any additional platform-specific configuration here
    }
}