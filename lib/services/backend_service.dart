import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class BackendService {
  // --- ENDPOINT CONFIG ---
  // If you deploy to Railway/Render, put your URL here (e.g., https://ocr-app.up.railway.app)
  static const String _cloudUrl = ''; 

  /// Dynamic Base URL: Prioritizes Cloud URL, then 10.0.2.2 (Android) or 127.0.0.1 (Others)
  String get _baseUrl {
    if (_cloudUrl.isNotEmpty) return _cloudUrl;
    
    if (Platform.isAndroid && !Platform.environment.containsKey('FLUTTER_TEST')) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://127.0.0.1:8000';
  }
  
  final http.Client _client = http.Client();

  /// Check if backend is alive
  Future<bool> checkHealth() async {
    try {
      final response = await _client.get(Uri.parse('$_baseUrl/health')).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Upload image to Python backend for Hybrid AI Extraction
  /// Returns JSON Map on success, or null on failure.
  Future<Map<String, dynamic>?> performOCR(File imageFile) async {
    final url = Uri.parse('$_baseUrl/ocr');
    
    print('[BACKEND_REQUEST] Hybrid AI Request to: $url');
    
    try {
      final request = http.MultipartRequest('POST', url)
        ..files.add(await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
        ));

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 300), // 5 min timeout for Model Download
      );
      
      final response = await http.Response.fromStream(streamedResponse);
      
      print('[BACKEND_RESPONSE] Status: ${response.statusCode}');
      
      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        
        // Backend now returns the FULL AI Object directly
        if (data.containsKey('error') || data['status'] == 'failed') {
          print('[BACKEND_ERROR] Server reported failure: ${data['reason'] ?? data['error']}');
          return null;
        }

        return data;
      } else {
        print('[BACKEND_ERROR] HTTP Error: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('[BACKEND_EXCEPTION] Connection Failed: $e');
      return null;
    }
  }

  void dispose() {
    _client.close();
  }
}
