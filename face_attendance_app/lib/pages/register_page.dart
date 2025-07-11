import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'dart:typed_data';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class DialogBox extends StatelessWidget {
  final TextEditingController nameController = TextEditingController();

  DialogBox({super.key});

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Enter Details'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: nameController,
            decoration: const InputDecoration(hintText: 'Name'),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () {
            final name = nameController.text;
            Navigator.of(context).pop({'name': name});
          },
          child: const Text('OK'),
        ),
      ],
    );
  }
}

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  bool _isLoading = false;
  XFile? _image;
  Uint8List? _compressedBytes;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final pickedImage = await _picker.pickImage(source: source);
    if (pickedImage != null) {
      final compressed = await FlutterImageCompress.compressWithFile(
        pickedImage.path,
        quality: 95,
        rotate: 0, // disable auto-rotation by forcing zero
        format: CompressFormat.jpeg,
      );

      if (compressed != null) {
        setState(() {
          _image = pickedImage;
          _compressedBytes = compressed;
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Image compression failed')),
        );
      }
    }
  }

  Future<void> _sendToApi(String name) async {
    if (_compressedBytes == null) return;

    setState(() {
      _isLoading = true;
    });

    final uri = Uri.parse('http://192.168.1.4:8000/register/register-student');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      http.MultipartFile.fromBytes(
        'files',
        _compressedBytes!,
        filename: 'captured.jpg',
        contentType: MediaType('image', 'jpeg'),
      ),
    );

    request.fields['student_name'] = name;

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Face registered successfully!')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to register face. ${response.body}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }

    setState(() {
      _isLoading = false;
      _image = null;
      _compressedBytes = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text('Register Person'),
          backgroundColor: Colors.blue,
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _compressedBytes == null
                  ? const Text('No image selected.')
                  : Image.memory(_compressedBytes!),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () => _pickImage(ImageSource.camera),
                child: const Text('Pick Image from Camera'),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () => _pickImage(ImageSource.gallery),
                child: const Text('Pick Image from Gallery'),
              ),
              const SizedBox(height: 20),
              _isLoading
                  ? const CircularProgressIndicator()
                  : ElevatedButton(
                      onPressed: () async {
                        final result = await showDialog(
                          context: context,
                          builder: (BuildContext context) {
                            return DialogBox();
                          },
                        );
                        if (result != null) {
                          final name = result['name'];
                          await _sendToApi(name);
                        }
                      },
                      child: const Text('Register Person'),
                    ),
            ],
          ),
        ));
  }
}
