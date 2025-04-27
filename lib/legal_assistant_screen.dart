import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';
// import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:file_picker/file_picker.dart';

void main() async {
  await dotenv.load(fileName: ".env");
  runApp(const LegalAssistantApp());
}

class LegalAssistantApp extends StatelessWidget {
  const LegalAssistantApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Know Your Rights India',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      darkTheme: ThemeData.dark().copyWith(
        primaryColor: Colors.blue,
        useMaterial3: true,
      ),
      themeMode: ThemeMode.system,
      home: const LegalAssistantScreen(),
    );
  }
}

class LegalAssistantScreen extends StatefulWidget {
  final String? pdfPath;

  const LegalAssistantScreen({Key? key, this.pdfPath}) : super(key: key);

  @override
  _LegalAssistantScreenState createState() => _LegalAssistantScreenState();
}

class _LegalAssistantScreenState extends State<LegalAssistantScreen> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _queryController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final VectorDBService _vectorDBService = VectorDBService();
  // final GeminiService _geminiService = GeminiService();
  final DeepInfraService _deepInfraService = DeepInfraService();
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _addWelcomeMessage();
    if (widget.pdfPath != null) {
      _processPdf(widget.pdfPath!);
    }
  }

  @override
  void dispose() {
    _queryController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _addWelcomeMessage() {
    setState(() {
      _messages.add(
        const ChatMessage(
          text:
              'Welcome to the Know Your Rights (KYR) legal assistant for India. '
              'I can help you understand your legal rights, explain legal processes, '
              'and provide information about laws in India. How can I assist you today?',
          isUser: false,
        ),
      );
    });
  }

  Future<void> _processPdf(String path) async {
    setState(() => _isLoading = true);
    try {
      // Implement your PDF processing logic here
      // Example: Send to backend for analysis
      var request = http.MultipartRequest(
          'POST', Uri.parse('${dotenv.env['BACKEND_URL']}/query'));
      request.files.add(await http.MultipartFile.fromPath('file', path));
      var response = await request.send();

      if (response.statusCode == 200) {
        setState(() {
          _messages.add(ChatMessage(
            text: "PDF document processed successfully and ready for queries",
            isUser: false,
          ));
        });
      } else {
        throw Exception('Failed to process PDF');
      }
    } catch (e) {
      setState(() {
        _messages.add(ChatMessage(
          text: "Error processing PDF: ${e.toString()}",
          isUser: false,
        ));
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // Future<void> _handleQuery(String text) async {
  //   _queryController.clear();
  //   if (text.trim().isEmpty) return;

  //   setState(() {
  //     _messages.add(ChatMessage(text: text, isUser: true));
  //     _isLoading = true;
  //   });
  //   _scrollToBottom();

  //   try {
  //     final vectorResults = await _vectorDBService.queryVectorDB(text, k: 3);
  //     final answer = await _geminiService.generateAnswer(text, vectorResults);

  //     setState(() {
  //       _messages.add(
  //         ChatMessage(text: answer, isUser: false, sources: vectorResults),
  //       );
  //       _isLoading = false;
  //     });
  //   } catch (e) {
  //     setState(() {
  //       _messages.add(
  //         ChatMessage(
  //           text: 'Sorry, I encountered an error: ${e.toString()}. Please try again.',
  //           isUser: false,
  //         ),
  //       );
  //       _isLoading = false;
  //     });
  //   }
  //   _scrollToBottom();
  // }

  Future<void> _handleQuery(String text) async {
    _queryController.clear();
    if (text.trim().isEmpty) return;

    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isLoading = true;
    });
    _scrollToBottom();

    try {
      final vectorResults = await _vectorDBService.queryVectorDB(text,
          k: 2); // Reduced from 3 to 2
      final answer = await _deepInfraService.generateLegalAnswer(
          text, vectorResults ?? []);

      setState(() {
        _messages.add(
          ChatMessage(text: answer, isUser: false, sources: vectorResults),
        );
      });
    } catch (e) {
      setState(() {
        _messages.add(
          ChatMessage(
            text:
                'Error: ${e.toString().replaceAll(RegExp(r'^Exception: '), '')}',
            isUser: false,
          ),
        );
      });
    } finally {
      setState(() => _isLoading = false);
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _uploadPdf() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );

    if (result != null && result.files.single.path != null) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => LegalAssistantScreen(
            pdfPath: result.files.single.path!,
          ),
        ),
      );
    }
  }

  void _showAboutDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About KYR India'),
        content: const SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Know Your Rights (KYR) India',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              SizedBox(height: 8),
              Text(
                'This application provides a legal assistant chatbot to help you '
                'understand your rights and legal processes in India using Gemini AI '
                'and a vector database.',
              ),
              SizedBox(height: 16),
              Text(
                'Disclaimer: This app provides general legal information and is not '
                'a substitute for professional legal advice.',
                style: TextStyle(fontStyle: FontStyle.italic),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Widget _buildChatInterface() {
    return Column(
      children: [
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
          child: Row(
            children: [
              ActionChip(
                label: const Text('Constitution'),
                onPressed: () => _handleQuery(
                  'What are my fundamental rights in the Indian Constitution?',
                ),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Criminal Rights'),
                onPressed: () => _handleQuery(
                  'What should I know if I am arrested in India?',
                ),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('RTI'),
                onPressed: () =>
                    _handleQuery('How can I file an RTI application?'),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Consumer Rights'),
                onPressed: () => _handleQuery(
                  'What are my rights as a consumer in India?',
                ),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Labor Laws'),
                onPressed: () => _handleQuery(
                  'What are the basic rights of workers in India?',
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            controller: _scrollController,
            padding: const EdgeInsets.all(8.0),
            itemCount: _messages.length,
            itemBuilder: (_, int index) => _messages[index],
          ),
        ),
        if (_isLoading)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8.0),
            child: LinearProgressIndicator(),
          ),
        Container(
          decoration: BoxDecoration(
            color: Theme.of(context).cardColor,
            boxShadow: [
              BoxShadow(
                blurRadius: 3,
                color: Colors.grey.withOpacity(0.2),
                offset: const Offset(0, -2),
              ),
            ],
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8.0),
          child: Row(
            children: [
              IconButton(
                icon: const Icon(Icons.upload_file),
                onPressed: _uploadPdf,
                tooltip: 'Upload PDF',
              ),
              Expanded(
                child: TextField(
                  controller: _queryController,
                  onSubmitted: _handleQuery,
                  decoration: const InputDecoration(
                    hintText: 'Ask about your rights in India...',
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.all(12),
                  ),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.send),
                onPressed: _isLoading
                    ? null
                    : () => _handleQuery(_queryController.text),
              ),
            ],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Know Your Rights - India'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showAboutDialog,
          ),
        ],
      ),
      body: _buildChatInterface(),
    );
  }
}

class ChatMessage extends StatelessWidget {
  final String text;
  final bool isUser;
  final List<Map<String, dynamic>>? sources;

  const ChatMessage({
    Key? key,
    required this.text,
    required this.isUser,
    this.sources,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: isUser
            ? Theme.of(context).primaryColor.withOpacity(0.2)
            : Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(16.0),
        border: Border.all(
          color: isUser
              ? Theme.of(context).primaryColor
              : Colors.grey.withOpacity(0.3),
          width: 1.0,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            isUser ? 'You' : 'Legal Assistant',
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: isUser ? Theme.of(context).primaryColor : Colors.grey[700],
            ),
          ),
          const SizedBox(height: 4.0),
          Text(text, style: const TextStyle(fontSize: 16.0)),
          if (!isUser && sources != null && sources!.isNotEmpty) ...[
            const SizedBox(height: 8.0),
            ExpansionTile(
              title: const Text(
                'View Sources',
                style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
              ),
              children: sources!
                  .map(
                    (source) => ListTile(
                      title: Text(source['source']),
                      subtitle: Text(source['text']),
                    ),
                  )
                  .toList(),
            ),
          ],
        ],
      ),
    );
  }
}

class VectorDBService {
  final String _baseUrl = dotenv.env['BACKEND_URL']!;

  Future<List<Map<String, dynamic>>> queryVectorDB(
    String query, {
    int k = 3,
  }) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/query'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': query, 'k': k}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return List<Map<String, dynamic>>.from(data['results']);
    }
    throw Exception('Failed to query vector database');
  }
}

// class GeminiService {
//   late final GenerativeModel _model;

//   GeminiService() {
//     final apiKey = dotenv.env['GEMINI_API_KEY']!;
//     _model = GenerativeModel(
//       model: 'gemini-1.5-pro',
//       apiKey: apiKey,
//       generationConfig: GenerationConfig(
//         maxOutputTokens: 50,
//         temperature: 0.7,
//       ),
//       safetySettings: [
//         SafetySetting(HarmCategory.harassment, HarmBlockThreshold.high),
//       ],
//     );
//   }

//   Future<String> generateAnswer(
//     String query,
//     List<Map<String, dynamic>> context,
//   ) async {
//     final prompt = '''
//     You are a legal expert assistant. Answer the user's question using ONLY the provided
//     legal documents context. If the answer isn't in the context, say you don't know.

//     Question: $query

//     Context:
//     ${context.map((doc) => "From ${doc['source']}:\n${doc['text']}").join('\n\n')}

//     Provide a concise, professional answer with references to the source documents.
//     ''';

//     final content = [Content.text(prompt)];
//     final response = await _model.generateContent(content);
//     return response.text ?? 'No answer could be generated.';
//   }
// }

class DeepInfraService {
  final String _apiKey = dotenv.env['DEEPINFRA_API_KEY'] ?? '';
  static const String _apiUrl =
      'https://api.deepinfra.com/v1/inference/mistralai/Mistral-7B-Instruct-v0.1';

  Future<String> generateLegalAnswer(
      String query, List<Map<String, dynamic>> context) async {
    if (_apiKey.isEmpty) {
      throw Exception('DeepInfra API key not configured');
    }

    // Prepare context (limit to 500 chars to avoid token limits)
    final contextText = context
        .take(2) // Only use top 2 relevant documents
        .map((doc) => "${doc['source']}: ${doc['text']}")
        .join('\n\n')
        .substring(0, 500);

    final prompt = '''
    [INST] <<SYS>>
    You are an expert legal assistant specializing in Indian law. 
    Answer concisely using the provided context when available.
    If unsure, say "I recommend consulting a qualified lawyer."
    <</SYS>>

    Context:
    $contextText

    Question: $query

    Provide a clear answer with relevant Indian legal provisions: [/INST]
    ''';

    try {
      final response = await http.post(
        Uri.parse(_apiUrl),
        headers: {
          'Authorization': 'Bearer $_apiKey',
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'input': prompt,
          'max_new_tokens': 350, // Enough for detailed legal answers
          'temperature': 0.3, // Keep responses factual
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['results'][0]['generated_text'].trim();
      } else {
        throw Exception('API Error: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('DeepInfra request failed: $e');
    }
  }
}
