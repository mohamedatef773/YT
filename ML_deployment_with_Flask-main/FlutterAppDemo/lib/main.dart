import 'package:xxxx/htttp.dart';

import 'package:flutter/material.dart';



Future   main() async {
  WidgetsFlutterBinding.ensureInitialized();
   
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(),

      home: Test1()
    );
  }
}

