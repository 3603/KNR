import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:url_launcher/url_launcher.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_selector/file_selector.dart';
import 'voice_service.dart'; // Add this import
import 'pdf_service.dart';
import 'legal_resources.dart';
// import 'package:pdfx/pdfx.dart';
import 'package:file_picker/file_picker.dart';
import 'rag_query_screen.dart';
import 'legal_assistant_screen.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'legal_news_screen.dart';
import 'GST_Calculator.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env"); // Load env before running app
  runApp(const LegalAwarenessApp());
}

class LegalAwarenessApp extends StatelessWidget {
  const LegalAwarenessApp({Key? key}) : super(key: key);

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
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final VoiceService _voiceService = VoiceService(); // Corrected capitalization
  final List<ChatMessage> _messages = [];
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  bool _isLoading = false;
  bool _showChatInterface = false;
  List<String> _uploadedPdfs = [];
  String _apiKey = "";
  final String _apiEndpoint =
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent";

  @override
  void initState() {
    super.initState();
    _loadApiKey();
    _loadUploadedPdfs();
    _addWelcomeMessage();
  }

  @override
  void dispose() {
    _voiceService.dispose();
    _textController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _addWelcomeMessage() {
    setState(() {
      _messages.add(
        ChatMessage(
          text:
              "Welcome to the Know Your Rights (KYR) legal assistant for India. I can help you understand your legal rights, explain legal processes, and provide information about laws in India. How can I assist you today?",
          isUser: false,
        ),
      );
    });
  }

  Future<void> _loadApiKey() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _apiKey = prefs.getString('api_key') ?? "";
    });
  }

  Future<void> _saveApiKey(String key) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('api_key', key);
    setState(() {
      _apiKey = key;
    });
  }

  Future<void> _loadUploadedPdfs() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _uploadedPdfs = prefs.getStringList('uploaded_pdfs') ?? [];
    });
  }

  Future<void> _saveUploadedPdfs() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('uploaded_pdfs', _uploadedPdfs);
  }

  void _handleSubmitted(String text) async {
    _textController.clear();
    if (text.trim().isEmpty) return;

    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isLoading = true;
    });

    _scrollToBottom();

    if (_apiKey.isEmpty) {
      setState(() {
        _messages.add(
          ChatMessage(
            text:
                "Please set your API key in the settings to use the AI assistant.",
            isUser: false,
          ),
        );
        _isLoading = false;
      });
      _scrollToBottom();
      return;
    }

    try {
      String systemPrompt = """
You are a legal assistant specializing in Indian law and citizens' rights. 
Your purpose is to provide accurate information about laws, rights, and legal processes in India.
Focus on providing factual, up-to-date legal information from reliable sources including:
1. The Constitution of India
2. Indian Legal Codes (IPC, CPC, CrPC, etc.)
3. Recent Supreme Court and High Court judgments
4. Government websites and official legal resources

Always cite your sources when providing information. If you're unsure about something, acknowledge the limitations of your knowledge and suggest seeking professional legal advice.
      """;

      if (_uploadedPdfs.isNotEmpty) {
        systemPrompt +=
            "\n\nThe user has uploaded these legal documents that you can reference: ${_uploadedPdfs.join(', ')}";
      }

      systemPrompt +=
          "\n\nYou should suggest searching government websites for the most current information when appropriate.";

      List<Map<String, dynamic>> conversationHistory = getRecentMessages();

      List<Map<String, dynamic>> contents = [
        {
          "role": "user",
          "parts": [
            {"text": systemPrompt}
          ]
        },
        {
          "role": "model",
          "parts": [
            {
              "text":
                  "I understand. I am a legal assistant specializing in Indian law and citizens' rights. I will provide accurate information and cite sources."
            }
          ]
        }
      ];

      for (var message in conversationHistory) {
        contents.add({
          "role": message['role'] == 'user' ? "user" : "model",
          "parts": [
            {"text": message['content']}
          ]
        });
      }

      contents.add({
        "role": "user",
        "parts": [
          {"text": text}
        ]
      });

      final response = await http.post(
        Uri.parse("$_apiEndpoint?key=$_apiKey"),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          "contents": contents,
          "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 100,
          },
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final reply = data['candidates'][0]['content']['parts'][0]['text'];

        setState(() {
          _messages.add(ChatMessage(text: reply, isUser: false));
          _isLoading = false;
        });
      } else {
        throw Exception(
            'Failed to get response: ${response.statusCode} ${response.body}');
      }
    } catch (e) {
      setState(() {
        _messages.add(
          ChatMessage(
            text:
                "Sorry, I encountered an error: ${e.toString()}. Please try again or check your API settings.",
            isUser: false,
          ),
        );
        _isLoading = false;
      });
    }

    _scrollToBottom();
  }

  List<Map<String, String>> getRecentMessages() {
    final List<ChatMessage> recentMessages = _messages.length > 10
        ? _messages.sublist(_messages.length - 10)
        : _messages;

    return recentMessages
        .map((message) => {
              'role': message.isUser ? 'user' : 'assistant',
              'content': message.text,
            })
        .toList();
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

  Future<void> _processPdf() async {
    final file = await openFile(acceptedTypeGroups: [
      XTypeGroup(label: 'PDFs', extensions: ['pdf'])
    ]);
    if (file == null) return;

    setState(() => _isLoading = true);
    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('http://192.168.0.100:8000/process-pdf'));
      request.files.add(await http.MultipartFile.fromPath('file', file.path));
      var response = await request.send();

      if (!mounted) return; // <- Critical check

      if (response.statusCode == 200) {
        var data = jsonDecode(await response.stream.bytesToString());
        Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => PdfSummaryScreen(
                summary: data['summary'],
                redFlags: data['red_flags'],
              ),
            ));
      } else {
        if (!mounted) return; // <- Check again
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${response.statusCode}')),
        );
      }
    } catch (e) {
      if (!mounted) return; // <- Check here too
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed: $e')),
      );
    } finally {
      if (mounted) {
        // <- Safeguard setState
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _uploadPdf() async {
    try {
      final XTypeGroup pdfGroup = XTypeGroup(
        label: 'PDFs',
        extensions: ['pdf'],
      );

      final XFile? file = await openFile(
        acceptedTypeGroups: [pdfGroup],
      );

      if (file != null) {
        String fileName = file.name;
        setState(() {
          _uploadedPdfs.add(fileName);
        });

        await _saveUploadedPdfs();

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Uploaded: $fileName')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error uploading PDF: ${e.toString()}')),
      );
    }
  }

  Future<void> openPdf() async {
    final file = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );

    if (file != null) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => PdfViewerScreen(
            pdfPath: file.files.single.path!,
          ),
        ),
      );
    }
  }

  //  IMPORTANT
  //   Future<void> _showPdf(String path) async {
  //   try {
  //     // 1. FIRST await the PDF document loading
  //     final Future<PdfDocument> pdfFuture = PdfDocument.openFile(path);

  //     // 2. THEN create the controller with the future
  //     final PdfController controller = PdfController(
  //       document: pdfFuture, // Pass the Future directly
  //     );

  //     if (!mounted) return;

  //     // 3. Navigate to PDF viewer
  //     await Navigator.push(
  //       context,
  //       MaterialPageRoute(
  //         builder: (context) => Scaffold(
  //           appBar: AppBar(title: const Text('PDF Viewer')),
  //           body: PdfView(controller: controller),
  //         ),
  //       ),
  //     );
  //   } catch (e) {
  //     if (!mounted) return;
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       SnackBar(content: Text('Failed to open PDF: ${e.toString()}')),
  //     );
  //   }
  // }

  void _openGovernmentWebsite(String website) async {
    final Uri url = Uri.parse(website);
    if (!await launchUrl(url)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not launch $website')),
      );
    }
  }

  void _showApiKeyDialog() {
    final TextEditingController apiKeyController =
        TextEditingController(text: _apiKey);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Set API Key'),
        content: TextField(
          controller: apiKeyController,
          decoration: const InputDecoration(
            labelText: 'API Key',
            hintText: 'Enter your API key',
          ),
          obscureText: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              _saveApiKey(apiKeyController.text);
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('API key saved')),
              );
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
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
                'This application aims to improve legal literacy and access to justice in India by providing easy-to-understand information about legal rights, processes, and resources.',
              ),
              SizedBox(height: 16),
              Text(
                'Disclaimer: This app provides general legal information and is not a substitute for professional legal advice. For specific legal issues, please consult a qualified lawyer.',
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

  Widget _buildHomeScreen() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          const SizedBox(height: 24),
          const Text(
            'Know Your Rights India',
            style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          const Text(
            'Legal awareness for all citizens',
            style: TextStyle(fontSize: 16, color: Colors.grey),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),

          // Grid of features
          GridView.count(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            childAspectRatio: 1.0,
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
            children: [
              // Chat with Legal Assistant Card
              _buildFeatureCard(
                icon: Icons.chat_bubble_outline,
                title: 'Chat Assistant',
                description: 'Ask legal questions',
                color: Colors.blue,
                onTap: () {
                  setState(() => _showChatInterface = true);
                },
              ),

              // PDF Processor Card
              _buildFeatureCard(
                icon: Icons.picture_as_pdf,
                title: 'PDF Processor',
                description: 'Analyze legal documents',
                color: Colors.red,
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const PdfProcessingScreen()),
                  );
                },
              ),

              // Legal Resources Card
              _buildFeatureCard(
                icon: Icons.library_books,
                title: 'Legal Docs',
                description: 'Access Legal Documents',
                color: Colors.green,
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const LegalResourcesScreen()),
                  );
                },
              ),

              // RAG Model Card (NEW)
              _buildFeatureCard(
                icon: Icons.calculate,
                title: 'GST Calculator',
                description: 'Simple GST Calculator',
                color: const Color.fromARGB(255, 44, 60, 208),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const GSTCalculator()),
                  );
                },
              ),
              _buildFeatureCard(
                icon: Icons.smart_toy,
                title: 'AI Legal Help',
                description: 'Advanced legal AI assistant',
                color: Colors.purple,
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const LegalAssistantScreen()),
                  );
                },
              ),
              _buildFeatureCard(
                icon: Icons.new_releases,
                title: 'Legal News',
                description: 'Latest updates',
                color: Colors.orange,
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const LegalNewsScreen()),
                  );
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureCard({
    required IconData icon,
    required String title,
    required String description,
    required Color color,
    required VoidCallback onTap,
  }) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 40, color: color),
              const SizedBox(height: 16),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              Text(
                description,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ),
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
                label: const Text('Home'),
                avatar: const Icon(Icons.home, size: 16),
                onPressed: () {
                  setState(() {
                    _showChatInterface = false;
                  });
                },
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Constitution'),
                onPressed: () => _handleSubmitted(
                    'What are my fundamental rights in the Indian Constitution?'),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Criminal Rights'),
                onPressed: () => _handleSubmitted(
                    'What should I know if I am arrested in India?'),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('RTI'),
                onPressed: () =>
                    _handleSubmitted('How can I file an RTI application?'),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Consumer Rights'),
                onPressed: () => _handleSubmitted(
                    'What are my rights as a consumer in India?'),
              ),
              const SizedBox(width: 8),
              ActionChip(
                label: const Text('Labor Laws'),
                onPressed: () => _handleSubmitted(
                    'What are the basic rights of workers in India?'),
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
              _voiceService.buildVoiceInputButton(context, _textController),
              Expanded(
                child: TextField(
                  controller: _textController,
                  onSubmitted: _handleSubmitted,
                  decoration: const InputDecoration(
                    hintText: 'Ask about your rights in India...',
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.all(12),
                  ),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.send),
                onPressed: () => _handleSubmitted(_textController.text),
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
            icon: const Icon(Icons.upload_file),
            onPressed: _uploadPdf,
            tooltip: 'Upload Legal Document',
          ),
          PopupMenuButton<String>(
            onSelected: (value) {
              switch (value) {
                case 'api_key':
                  _showApiKeyDialog();
                  break;
                case 'about':
                  _showAboutDialog();
                  break;
                case 'e_courts':
                  _openGovernmentWebsite('https://ecourts.gov.in/');
                  break;
                case 'india_code':
                  _openGovernmentWebsite('https://www.indiacode.nic.in/');
                  break;
                case 'legal_services':
                  _openGovernmentWebsite('https://nalsa.gov.in/');
                  break;
              }
            },
            itemBuilder: (BuildContext context) => [
              const PopupMenuItem(
                value: 'api_key',
                child: Text('Set API Key'),
              ),
              const PopupMenuItem(
                value: 'about',
                child: Text('About'),
              ),
              const PopupMenuDivider(),
              const PopupMenuItem(
                value: 'e_courts',
                child: Text('e-Courts'),
              ),
              const PopupMenuItem(
                value: 'india_code',
                child: Text('India Code'),
              ),
              const PopupMenuItem(
                value: 'legal_services',
                child: Text('Legal Services'),
              ),
            ],
          ),
        ],
      ),
      body: _showChatInterface ? _buildChatInterface() : _buildHomeScreen(),
    );
  }
}

class ChatMessage extends StatelessWidget {
  final String text;
  final bool isUser;

  const ChatMessage({Key? key, required this.text, required this.isUser})
      : super(key: key);

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
          Row(
            children: [
              Expanded(
                child: Text(
                  text,
                  style: const TextStyle(fontSize: 16.0),
                ),
              ),
              if (!isUser) VoiceService().buildTtsButton(context, text),
            ],
          ),
          if (!isUser) _buildReferenceLinks(context, text),
        ],
      ),
    );
  }

  Widget _buildReferenceLinks(BuildContext context, String text) {
    final RegExp regExp = RegExp(r'(https?://[^\s]+\.gov\.in[^\s]*)');
    final matches = regExp.allMatches(text);

    if (matches.isEmpty) {
      return Container();
    }

    return Padding(
      padding: const EdgeInsets.only(top: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'References:',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14.0),
          ),
          const SizedBox(height: 4.0),
          ...matches.map((match) {
            final url = match.group(0)!;
            return Padding(
              padding: const EdgeInsets.only(top: 4.0),
              child: InkWell(
                onTap: () async {
                  final Uri uri = Uri.parse(url);
                  if (await canLaunchUrl(uri)) {
                    await launchUrl(uri);
                  }
                },
                child: Text(
                  url,
                  style: TextStyle(
                    color: Theme.of(context).primaryColor,
                    decoration: TextDecoration.underline,
                    fontSize: 14.0,
                  ),
                ),
              ),
            );
          }).toList(),
        ],
      ),
    );
  }
}

class RightsCategory {
  final String title;
  final String description;
  final String icon;
  final List<String> commonQuestions;

  RightsCategory({
    required this.title,
    required this.description,
    required this.icon,
    required this.commonQuestions,
  });
}

class PdfSummaryScreen extends StatelessWidget {
  final String summary;
  final Map<String, dynamic> redFlags;

  const PdfSummaryScreen(
      {Key? key, required this.summary, required this.redFlags})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Document Summary')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Summary:',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 8),
            Text(summary),
            const SizedBox(height: 20),
            if (redFlags.isNotEmpty) ...[
              const Text('⚠️ Red Flags:',
                  style: TextStyle(
                      color: Colors.red, fontWeight: FontWeight.bold)),
              ...redFlags.entries.map((e) => Text("• ${e.key}: ${e.value}")),
            ],
          ],
        ),
      ),
    );
  }
}

class RightsCategoryList extends StatelessWidget {
  final void Function(String) onQuestionTap;

  RightsCategoryList({Key? key, required this.onQuestionTap}) : super(key: key);

  final List<RightsCategory> categories = [
    RightsCategory(
      title: 'Fundamental Rights',
      description: 'Basic rights guaranteed by the Indian Constitution',
      icon: '🏛️',
      commonQuestions: [
        'What are my fundamental rights in India?',
        'How can I file a petition for violation of fundamental rights?',
        'What is Article 21 of the Indian Constitution?',
      ],
    ),
    RightsCategory(
      title: 'Criminal Rights',
      description: 'Rights of accused persons and victims in criminal cases',
      icon: '⚖️',
      commonQuestions: [
        'What are my rights if I am arrested?',
        'Can the police search my home without a warrant?',
        'How do I file an FIR?',
      ],
    ),
    RightsCategory(
      title: 'Consumer Rights',
      description: 'Protection against unfair trade practices',
      icon: '🛒',
      commonQuestions: [
        'How do I file a consumer complaint?',
        'What remedies are available under Consumer Protection Act?',
        'What are my rights when purchasing goods online?',
      ],
    ),
    RightsCategory(
      title: 'Labor Rights',
      description: 'Rights of workers and employees',
      icon: '👷',
      commonQuestions: [
        'What is the minimum wage in my state?',
        'How many hours can I legally work?',
        'What maternity benefits am I entitled to?',
      ],
    ),
    RightsCategory(
      title: 'RTI',
      description: 'Right to Information Act and its applications',
      icon: '📄',
      commonQuestions: [
        'How do I file an RTI application?',
        'What information can I request through RTI?',
        'What are the fees for filing RTI?',
      ],
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: categories.length,
      itemBuilder: (context, index) {
        final category = categories[index];
        return Card(
          margin: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
          elevation: 2.0,
          child: ExpansionTile(
            leading: Text(
              category.icon,
              style: const TextStyle(fontSize: 24.0),
            ),
            title: Text(
              category.title,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            subtitle: Text(category.description),
            children: [
              ...category.commonQuestions.map((question) => ListTile(
                    title: Text(question),
                    trailing: const Icon(Icons.arrow_forward_ios, size: 16.0),
                    onTap: () => onQuestionTap(question),
                  )),
            ],
          ),
        );
      },
    );
  }
}

class DocumentViewer extends StatelessWidget {
  final String documentTitle;

  const DocumentViewer({Key? key, required this.documentTitle})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(documentTitle),
      ),
      body: const Center(
        child: Text('Document viewer would display the content here'),
      ),
    );
  }
}

class OnboardingScreen extends StatefulWidget {
  final VoidCallback onComplete;

  const OnboardingScreen({Key? key, required this.onComplete})
      : super(key: key);

  @override
  _OnboardingScreenState createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  final List<Map<String, String>> _pages = [
    {
      'title': 'Welcome to Know Your Rights',
      'description':
          'Your digital legal assistant to help you understand your rights and legal processes in India.',
      'image': '🏛️',
    },
    {
      'title': 'Ask Legal Questions',
      'description':
          'Get information about your rights, laws, and legal procedures through our AI-powered chat interface.',
      'image': '💬',
    },
    {
      'title': 'Upload Documents',
      'description':
          'Upload legal documents for reference and get assistance understanding them.',
      'image': '📄',
    },
    {
      'title': 'Access Government Resources',
      'description':
          'Direct links to official government portals for authentic legal information.',
      'image': '🔗',
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                controller: _pageController,
                onPageChanged: (int page) {
                  setState(() {
                    _currentPage = page;
                  });
                },
                itemCount: _pages.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.all(32.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          _pages[index]['image']!,
                          style: const TextStyle(fontSize: 80.0),
                        ),
                        const SizedBox(height: 32.0),
                        Text(
                          _pages[index]['title']!,
                          style: const TextStyle(
                            fontSize: 24.0,
                            fontWeight: FontWeight.bold,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 16.0),
                        Text(
                          _pages[index]['description']!,
                          style: const TextStyle(fontSize: 16.0),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: List.generate(
                      _pages.length,
                      (index) => Container(
                        margin: const EdgeInsets.symmetric(horizontal: 4.0),
                        height: 8.0,
                        width: 8.0,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _currentPage == index
                              ? Theme.of(context).primaryColor
                              : Colors.grey.withOpacity(0.3),
                        ),
                      ),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: _currentPage == _pages.length - 1
                        ? widget.onComplete
                        : () {
                            _pageController.nextPage(
                              duration: const Duration(milliseconds: 300),
                              curve: Curves.easeInOut,
                            );
                          },
                    child: Text(
                      _currentPage == _pages.length - 1
                          ? 'Get Started'
                          : 'Next',
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class LegalAssistantApp extends StatefulWidget {
  @override
  _LegalAssistantAppState createState() => _LegalAssistantAppState();
}

class _LegalAssistantAppState extends State<LegalAssistantApp> {
  final TextEditingController _queryController = TextEditingController();
  String _response = '';
  bool _isLoading = false;

  Future<void> _sendQuery() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final response = await http.post(
        Uri.parse('http://192.168.0.100:8000/query'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': _queryController.text}),
      );

      if (response.statusCode == 200) {
        setState(() {
          _response = jsonDecode(response.body)['answer'];
        });
      } else {
        throw Exception('Failed to get response');
      }
    } catch (e) {
      setState(() {
        _response = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _uploadPdf() async {
    // Your existing file picker implementation
    // Then send to /process-pdf endpoint
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Legal RAG Assistant')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _queryController,
              decoration: InputDecoration(
                labelText: 'Ask a legal question',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isLoading ? null : _sendQuery,
              child:
                  _isLoading ? CircularProgressIndicator() : Text('Get Answer'),
            ),
            SizedBox(height: 16),
            Expanded(
              child: SingleChildScrollView(
                padding: EdgeInsets.all(8),
                child: Text(
                  _response,
                  style: TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _uploadPdf,
        tooltip: 'Upload PDF',
        child: Icon(Icons.upload),
      ),
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _checkFirstLaunch();
  }

  Future<void> _checkFirstLaunch() async {
    final prefs = await SharedPreferences.getInstance();
    final bool isFirstLaunch = prefs.getBool('first_launch') ?? true;

    Future.delayed(const Duration(seconds: 2), () {
      if (isFirstLaunch) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => OnboardingScreen(
              onComplete: () async {
                await prefs.setBool('first_launch', false);
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => const HomePage()),
                );
              },
            ),
          ),
        );
      } else {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const HomePage()),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              '🏛️',
              style: TextStyle(fontSize: 80.0),
            ),
            const SizedBox(height: 24.0),
            Text(
              'Know Your Rights',
              style: TextStyle(
                fontSize: 28.0,
                fontWeight: FontWeight.bold,
                color: Theme.of(context).primaryColor,
              ),
            ),
            const SizedBox(height: 8.0),
            const Text(
              'Legal Awareness for All',
              style: TextStyle(fontSize: 16.0),
            ),
            const SizedBox(height: 32.0),
            const CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }
}
