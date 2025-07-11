import 'package:flutter/material.dart'; //contains runApp function
import 'package:face_attendance_app/pages/register_page.dart'; //importing the register page
import 'package:face_attendance_app/pages/detect_page.dart'; //importing the detection page

void main() {
  runApp(
      //runApp is the built-in function and MyApp is the widget (parameter of the function)
      const MaterialApp(
    title: 'face attendance demo',
    home: SafeArea(
        child:
            HomeScreen()), //text is actually a built in widet. im using it as a placeholder
  ));
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});
  @override //because we're overrinding the build method in the statelesswidget class
  Widget build(BuildContext context) {
    return Material(
        child: Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text(
            'Face Attendance Demo',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(
            height: 20,
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const RegisterPage()));
            },
            child: const Text('Register Person'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.push(context,
                  MaterialPageRoute(builder: (context) => const DetectPage()));
            },
            child: const Text('Record Attendance'),
          ),
        ],
      ),
    ));
  }
}
//MyApp will initialize the widget tree
