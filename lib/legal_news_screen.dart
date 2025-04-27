import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:url_launcher/url_launcher.dart';
// import 'package:flutter_dotenv/flutter_dotenv.dart';

class LegalNewsScreen extends StatefulWidget {
  const LegalNewsScreen({Key? key}) : super(key: key);
  @override
  // Remove the asterisks
  LegalNewsScreenState createState() => LegalNewsScreenState();
}

class LegalNewsScreenState extends State<LegalNewsScreen> {
  List<dynamic> _newsArticles = [];
  bool _isLoading = true;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _fetchLegalNews();
  }

  Future<void> _fetchLegalNews() async {
    final apiKey = ""; // add your API key here
    print("Using key: $apiKey");

    try {
      // Add country parameter which is required for top-headlines
      final response = await http.get(
        Uri.parse(
            'https://newsapi.org/v2/everything?q=Court+india&apiKey=$apiKey'),
      );

      // Print the full response for debugging
      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body);

        // Check status before accessing articles
        if (responseData['status'] == 'ok') {
          if (responseData['articles'] != null &&
              responseData['articles'].isNotEmpty) {
            setState(() {
              _newsArticles = responseData['articles'];
              _isLoading = false;
            });
          } else {
            setState(() {
              _errorMessage = 'No articles found';
              _isLoading = false;
            });
          }
        } else {
          throw Exception(
              responseData['message'] ?? 'API returned error status');
        }
      } else {
        throw Exception('Failed to load news: ${response.statusCode}');
      }
    } catch (e) {
      print('Exception caught: $e');
      setState(() {
        _isLoading = false;
        _errorMessage = e.toString();
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Legal News Updates'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _errorMessage.isNotEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('Error: $_errorMessage'),
                      ElevatedButton(
                        onPressed: () {
                          setState(() {
                            _isLoading = true;
                            _errorMessage = '';
                          });
                          _fetchLegalNews();
                        },
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              : _newsArticles.isEmpty
                  ? const Center(child: Text('No news articles found'))
                  : ListView.builder(
                      itemCount: _newsArticles.length,
                      itemBuilder: (context, index) {
                        final article = _newsArticles[index];
                        return Card(
                          margin: const EdgeInsets.all(8),
                          child: ListTile(
                            leading: article['urlToImage'] != null
                                ? Image.network(
                                    article['urlToImage'],
                                    width: 50,
                                    errorBuilder: (ctx, obj, stk) =>
                                        const Icon(Icons.article),
                                  )
                                : const Icon(Icons.article),
                            title: Text(article['title'] ?? 'No title'),
                            subtitle: Text(
                              article['description'] ?? 'No description',
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                            onTap: () => _showNewsDetail(article),
                          ),
                        );
                      },
                    ),
    );
  }

  void _showNewsDetail(Map<String, dynamic> article) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(article['title']),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min, // Avoid excessive height
            children: [
              if (article['urlToImage'] != null)
                Image.network(
                  article['urlToImage'],
                  errorBuilder: (ctx, obj, stk) => const SizedBox.shrink(),
                ),
              const SizedBox(height: 16),
              Text(article['content'] ?? 'No content available'),
              const SizedBox(height: 16),
              Text(
                'Source: ${article['source']['name']}',
                style: TextStyle(color: Colors.grey[600]),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
          TextButton(
            onPressed: () => _launchURL(article['url']),
            child: const Text('Read Full'),
          ),
        ],
      ),
    );
  }

  // Updated to use the newer URL launcher API
  void _launchURL(String url) async {
    final Uri uri = Uri.parse(url);
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not launch $url')),
      );
    }
  }
}
