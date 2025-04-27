import 'package:flutter/material.dart';
import 'package:file_selector/file_selector.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class RagQueryScreen extends StatefulWidget {
  const RagQueryScreen({Key? key}) : super(key: key);

  @override
  _RagQueryScreenState createState() => _RagQueryScreenState();
}

class _RagQueryScreenState extends State<RagQueryScreen> {
  final TextEditingController _queryController = TextEditingController();
  String _response = '';
  bool _isLoading = false;
  List<String> _uploadedDocuments = [];

  @override
  void initState() {
    super.initState();
    _loadUploadedDocuments();
  }

  Future<void> _loadUploadedDocuments() async {
    // Load previously uploaded documents
    // Implement your storage logic here
  }

  Future<void> _uploadDocument() async {
    try {
      final XTypeGroup pdfGroup = XTypeGroup(label: 'PDFs', extensions: ['pdf']);
      final XFile? file = await openFile(acceptedTypeGroups: [pdfGroup]);
      
      if (file != null) {
        setState(() => _isLoading = true);
        
        var request = http.MultipartRequest(
          'POST', 
          Uri.parse('http://192.168.0.106:8000/process-pdf')
        );
        request.files.add(await http.MultipartFile.fromPath('file', file.path));
        
        var response = await request.send();
        if (response.statusCode == 200) {
          setState(() {
            _uploadedDocuments.add(file.name);
            _response = 'Document processed successfully!\nYou can now query its contents.';
          });
        } else {
          throw Exception('Failed to process document');
        }
      }
    } catch (e) {
      setState(() => _response = 'Error: ${e.toString()}');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _queryDocuments() async {
    if (_queryController.text.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _response = '';
    });

    try {
      final response = await http.post(
        Uri.parse('http://192.168.0.105:8000/query'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': _queryController.text}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() => _response = data['answer']);
      } else {
        throw Exception('Failed to get response');
      }
    } catch (e) {
      setState(() => _response = 'Error: ${e.toString()}');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Document AI Query'),
        actions: [
          IconButton(
            icon: const Icon(Icons.upload_file),
            onPressed: _uploadDocument,
            tooltip: 'Upload Document',
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            if (_uploadedDocuments.isNotEmpty) ...[
              const Text('Uploaded Documents:', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              SizedBox(
                height: 60,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: _uploadedDocuments.length,
                  itemBuilder: (context, index) {
                    return Padding(
                      padding: const EdgeInsets.only(right: 8.0),
                      child: Chip(
                        label: Text(_uploadedDocuments[index]),
                        deleteIcon: const Icon(Icons.close),
                        onDeleted: () {
                          setState(() {
                            _uploadedDocuments.removeAt(index);
                          });
                        },
                      ),
                    );
                  },
                ),
              ),
              const Divider(),
            ],
            TextField(
              controller: _queryController,
              decoration: InputDecoration(
                labelText: 'Ask about your documents',
                suffixIcon: IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _queryDocuments,
                ),
                border: const OutlineInputBorder(),
              ),
              maxLines: 3,
              minLines: 1,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _queryDocuments,
              child: const Text('Query Documents'),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: _isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : SingleChildScrollView(
                          child: Text(_response),
                        ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}