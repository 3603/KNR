import 'package:http/http.dart' as http;
import 'dart:convert';

class RagService {
  static const String _baseUrl = "http://192.168.0.105:8000";

  // Upload PDF
  static Future<String> uploadPdf(String filePath) async {
    var request = http.MultipartRequest(
      'POST', 
      Uri.parse('$_baseUrl/process-pdf')
    );
    request.files.add(
      await http.MultipartFile.fromPath('file', filePath)
    );
    
    final response = await request.send();
    return await response.stream.bytesToString();
  }

  // Query RAG Model
  static Future<String> askQuestion(String question) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/query'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'query': question}),
    );
    return jsonDecode(response.body)['answer'];
  }
}