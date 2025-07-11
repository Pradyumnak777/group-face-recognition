import 'dart:convert';
import 'dart:io';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:http_parser/http_parser.dart';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class DetectPage extends StatefulWidget {
  const DetectPage({super.key});

  @override
  State<DetectPage> createState() => _DetectPageState();
}

class _DetectPageState extends State<DetectPage> {
  XFile? _image;
  Uint8List? _annotatedImageBytes;
  List<String> _names = [];
  bool _isLoading = false;
  Uint8List? _compressedBytes;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickAndDetect({required ImageSource source}) async {
    final pickedImage = await _picker.pickImage(source: source);
    if (pickedImage == null) return;

    setState(() {
      _isLoading = true;
      _image = pickedImage;
    });

    final url = Uri.parse('http://192.168.1.4:8000/mark/detect');
    final request = http.MultipartRequest('POST', url);

    _compressedBytes = await FlutterImageCompress.compressWithFile(
      pickedImage.path,
      quality: 95,
      rotate: 0,
      format: CompressFormat.jpeg,
    );

    if (_compressedBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Failed to process image')),
      );
      setState(() {
        _isLoading = false;
      });
      return;
    }

    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        _compressedBytes!,
        filename: 'selected.jpg',
        contentType: MediaType('image', 'jpeg'),
      ),
    );

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        final base64Image = json['image'];
        final names = List<String>.from(json['names']);

        setState(() {
          _annotatedImageBytes = base64Decode(base64Image);
          _names = names;
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Detection failed: ${response.statusCode}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }

    setState(() {
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detect Faces'),
        backgroundColor: Colors.green,
      ),
      body: Center(
        child: _isLoading
            ? const CircularProgressIndicator()
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: () => _pickAndDetect(source: ImageSource.camera),
                    child: const Text('Capture & Detect (Camera)'),
                  ),
                  const SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: () =>
                        _pickAndDetect(source: ImageSource.gallery),
                    child: const Text('Select Image from Gallery'),
                  ),
                  const SizedBox(height: 20),
                  if (_annotatedImageBytes != null)
                    Column(
                      children: [
                        Image.memory(_annotatedImageBytes!),
                        const SizedBox(height: 10),
                        Text(
                          'Detected: ${_names.join(', ')}',
                          style: const TextStyle(fontSize: 16),
                        ),
                      ],
                    )
                  else if (_compressedBytes != null)
                    Image.memory(_compressedBytes!)
                  else
                    const Text('No image available.'),
                ],
              ),
      ),
    );
  }
}
