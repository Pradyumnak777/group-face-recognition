import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;

//camera access must be provided, after clicking, a preview for confirmation, and then an add button.
//api should be contacted via a buttono click, and the image should be sent to the api for processing.
//finally a message saying - 'added' should be displayed

class DialogBox extends StatelessWidget {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController groupController = TextEditingController();

  DialogBox({super.key});

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Enter Details'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: nameController, // Store Name input
            decoration: const InputDecoration(hintText: 'Name'),
          ),
          TextField(
            controller: groupController, // Store Group Name input
            decoration: const InputDecoration(hintText: 'Group Name'),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () {
            // Access the values using nameController.text and groupController.text
            final name = nameController.text;
            final group = groupController.text;

            // Pass the values back or use them as needed
            Navigator.of(context).pop({'name': name, 'group': group});
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
  XFile?
      _image; // holds the picked image, this can be used in the functions below (as a global variable)
  final ImagePicker _picker = ImagePicker(); // camera/gallery access

  // Step 1: Pick image from camera or gallery
  Future<void> _pickImage(ImageSource source) async {
    //ImageSource can be camera or gallery, its the arg here
    //this is a function, that will be called in the static part
    final pickedImage = await _picker.pickImage(
        source:
            source); //_picker.pickImage is a method that will open the camera or gallery
    if (pickedImage != null) {
      // if an image is picked
      setState(() {
        _image = pickedImage; // now '_image' holds a jpg/png file
      });
    }
  }

  // Step 2:  API call
  Future<void> _sendToApi(String name, String group) async {
    //first, take in name and group name
    if (_image == null) return;
    //send the _image to the server
    final uri = Uri.parse('http://your-api-url:8000/register-student');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath(
        'files', // field name in the API
        _image!.path, // path to the image file
      ),
    );

    request.fields['student_name'] = name;
    request.fields['group_name'] = group;

    final response = await request.send();
    if (response.statusCode == 200) {
      // Handle success
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Face registered successfully!')),
      );
    } else {
      // Handle error
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Failed to register face.')),
      );
    }

    // Step 3: Show success message

    // Optionally clear image after success
    setState(() {
      _image = null;
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
              _image == null
                  ? const Text('No image selected.')
                  : Image.file(File(_image!.path)), // display the image
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
              ElevatedButton(
                onPressed: () async {
                  //here, a popup should be shown, asking for name and group name
                  //then, the opo up should be closed and the image should be sent to the api
                  //after the image is sent, a message should be shown saying - 'added'
                  // DialogBox(); this wont work, so we need to use showDialog
                  final result = await showDialog(
                    context: context,
                    builder: (BuildContext context) {
                      return DialogBox();
                    },
                  );
                  if (result != null) {
                    // If the user provided input, send it to the API
                    final name = result['name'];
                    final group = result['group'];
                    await _sendToApi(name, group);
                  }
                },
                child: const Text('Register Person'),
              ),
            ],
          ),
        ));
  }
}
