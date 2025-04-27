import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:convert';
import 'package:flutter_pdfview/flutter_pdfview.dart';

class PdfViewerScreen extends StatelessWidget {
  final String pdfPath;

  const PdfViewerScreen({Key? key, required this.pdfPath}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Document Viewer')),
      body: PDFView(
        filePath: pdfPath,
        enableSwipe: true,
        swipeHorizontal: false,
        autoSpacing: true,
        onError: (error) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Failed to load PDF: $error")),
          );
        },
        onPageError: (page, error) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Failed to load page $page: $error")),
          );
        },
      ),
    );
  }
}

class LegalResourcesScreen extends StatefulWidget {
  const LegalResourcesScreen({Key? key}) : super(key: key);

  @override
  _LegalResourcesScreenState createState() => _LegalResourcesScreenState();
}

class _LegalResourcesScreenState extends State<LegalResourcesScreen> {
  List<String> _bundledResources = [];
  List<String> _userResources = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadResources();
  }

  Future<void> _loadResources() async {
    try {
      // Load bundled resources
      final manifestContent = await rootBundle.loadString('AssetManifest.json');
      final manifestMap = jsonDecode(manifestContent) as Map<String, dynamic>;
      
      setState(() {
        _bundledResources = manifestMap.keys
            .where((key) => key.endsWith('.pdf') && key.startsWith('assets/legal_resources/'))
            .toList();
      });

      // Load user-added resources
      final prefs = await SharedPreferences.getInstance();
      _userResources = prefs.getStringList('user_legal_resources') ?? [];
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error loading resources: ${e.toString()}")),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _saveUserResources() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('user_legal_resources', _userResources);
  }

  Future<void> _openPdf(String path, String title) async {
    try {
      String finalPath = path;
      
      if (path.startsWith('assets/')) {
        // For bundled resources
        final byteData = await rootBundle.load(path);
        final tempDir = await getTemporaryDirectory();
        final tempFile = File('${tempDir.path}/${DateTime.now().millisecondsSinceEpoch}_$title');
        await tempFile.writeAsBytes(byteData.buffer.asUint8List());
        finalPath = tempFile.path;
      }

      if (await File(finalPath).exists()) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => PdfViewerScreen(pdfPath: finalPath),
          ),
        );
      } else {
        throw Exception("File not found at $finalPath");
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error opening PDF: ${e.toString()}')),
      );
    }
  }

  Future<void> _uploadPdf() async {
    try {
      final XTypeGroup pdfGroup = XTypeGroup(
        label: 'PDFs',
        extensions: ['pdf'],
      );

      final XFile? file = await openFile(acceptedTypeGroups: [pdfGroup]);

      if (file != null && await File(file.path).exists()) {
        setState(() => _userResources.add(file.path));
        await _saveUserResources();
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Added: ${file.name}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error adding PDF: ${e.toString()}')),
      );
    }
  }

  Future<void> _deleteResource(int index) async {
    try {
      final path = _userResources[index];
      setState(() => _userResources.removeAt(index));
      await _saveUserResources();
      
      // Optionally delete the physical file
      final file = File(path);
      if (await file.exists()) {
        await file.delete();
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error deleting file: ${e.toString()}')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Legal Resources'),
        actions: [
          IconButton(
            icon: const Icon(Icons.upload_file),
            onPressed: _uploadPdf,
            tooltip: 'Add Legal Resource',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _buildContent(),
    );
  }

  Widget _buildContent() {
    if (_bundledResources.isEmpty && _userResources.isEmpty) {
      return const Center(
        child: Text(
          'No legal resources available.\nAdd some using the upload button.',
          textAlign: TextAlign.center,
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_bundledResources.isNotEmpty)
            _buildResourceSection('Bundled Resources', _bundledResources, isUserResource: false),
          if (_userResources.isNotEmpty)
            _buildResourceSection('Your Resources', _userResources, isUserResource: true),
        ],
      ),
    );
  }

  Widget _buildResourceSection(String title, List<String> resources, {required bool isUserResource}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 8.0, top: 16.0),
          child: Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        ListView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: resources.length,
          itemBuilder: (context, index) {
            final path = resources[index];
            final fileName = path.split('/').last;
            
            return Card(
              margin: const EdgeInsets.symmetric(vertical: 4.0),
              child: ListTile(
                leading: const Icon(Icons.picture_as_pdf, color: Colors.red),
                title: Text(fileName),
                onTap: () => _openPdf(path, fileName),
                trailing: isUserResource
                    ? IconButton(
                        icon: const Icon(Icons.delete),
                        onPressed: () => _deleteResource(index),
                      )
                    : null,
              ),
            );
          },
        ),
      ],
    );
  }
}