import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

class PdfService {
  Future<XFile?> pickPdfFile() async {
    try {
      final XTypeGroup pdfGroup = XTypeGroup(
        label: 'PDFs',
        extensions: ['pdf'],
      );

      final XFile? file = await openFile(
        acceptedTypeGroups: [pdfGroup],
      );
      
      return file;
    } catch (e) {
      print('Error picking PDF file: $e');
      return null;
    }
  }

  Future<String> extractTextFromApi(XFile file) async {
    try {
      final uri = Uri.parse("http://192.168.0.100:8000/process-pdf");
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', file.path));
      
      final response = await request.send();
      if (response.statusCode == 200) {
        return await response.stream.bytesToString();
      } else {
        throw Exception('Failed to process PDF: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('API Error: $e');
    }
  }
}

class PdfProcessingScreen extends StatefulWidget {
  const PdfProcessingScreen({Key? key}) : super(key: key);
  
  @override
  _PdfProcessingScreenState createState() => _PdfProcessingScreenState();
}

class _PdfProcessingScreenState extends State<PdfProcessingScreen> {
  final PdfService _pdfService = PdfService();
  bool _isLoading = false;
  String _loadingMessage = 'Processing...';
  XFile? _selectedFile;

  Future<void> _processPdf() async {
    if (_selectedFile == null) {
      final file = await _pdfService.pickPdfFile();
      if (file == null) return;
      setState(() => _selectedFile = file);
    }
    
    setState(() {
      _isLoading = true;
      _loadingMessage = 'Processing PDF...';
    });

    try {
      final text = await _pdfService.extractTextFromApi(_selectedFile!);
      // Handle the API response here
      print('Extracted text: $text');
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('PDF processed successfully')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PDF Processor'),
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_loadingMessage),
                ],
              ),
            )
          : Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Icon(
                    Icons.picture_as_pdf,
                    size: 80,
                    color: Colors.blue
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    'Legal Document Processor',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Upload a legal document to extract text',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 16, color: Colors.grey),
                  ),
                  const SizedBox(height: 32),
                  if (_selectedFile != null)
                    Card(
                      elevation: 2,
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Row(
                          children: [
                            const Icon(Icons.file_present),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _selectedFile!.name,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            IconButton(
                              icon: const Icon(Icons.close),
                              onPressed: () {
                                setState(() {
                                  _selectedFile = null;
                                });
                              },
                            ),
                          ],
                        ),
                      ),
                    ),
                  const SizedBox(height: 16),
                  ElevatedButton.icon(
                    icon: const Icon(Icons.file_upload),
                    label: const Text('Select PDF Document'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                    onPressed: () async {
                      final file = await _pdfService.pickPdfFile();
                      if (file != null) {
                        setState(() {
                          _selectedFile = file;
                        });
                      }
                    },
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton.icon(
                    icon: const Icon(Icons.auto_awesome_motion),
                    label: const Text('Process Document'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                    onPressed: _processPdf,
                  ),
                ],
              ),
            ),
    );
  }
}