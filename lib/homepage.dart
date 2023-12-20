import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_v2/tflite_v2.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? imageFile;
  final picker = ImagePicker();
  late bool _loading;

  @override
  void initState() {
    super.initState();
    _loading = false; // Initialize _loading
    // Load your TensorFlow Lite model here
    loadModel();
  }

  void loadModel() async {
    await Tflite.loadModel(
      model: 'assets/model.tflite', // Adjust the path to your model
      labels: 'assets/labels.txt', // Adjust the path to your labels
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Select Image'),
        ),
        body: Center(
          child: Column(
            children: [
              const SizedBox(
                height: 20.0,
              ),
              imageFile == null
                  ? Image.asset(
                      'assets/no_profile_image.png',
                      height: 300.0,
                      width: 300.0,
                    )
                  : Container(
                      height: 244.0,
                      width: 244.0,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(
                          12.0,
                        ), // You can adjust the border radius
                        image: DecorationImage(
                          image: FileImage(imageFile!),
                          fit: BoxFit.fill, // Adjust the BoxFit as needed
                        ),
                      ),
                    ),
              const SizedBox(
                height: 20.0,
              ),
              ElevatedButton(
                onPressed: () async {
                  Map<Permission, PermissionStatus> statuses =
                      await [Permission.storage, Permission.camera].request();
                  if (statuses[Permission.storage]!.isGranted &&
                      statuses[Permission.camera]!.isGranted) {
                    showImagePicker(context);
                  } else {
                    print('No permission provided');
                  }
                },
                child: Text('Select Image'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void showImagePicker(BuildContext context) {
    showModalBottomSheet(
      context: context,
      builder: (builder) {
        return Card(
          child: Container(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height / 5.2,
            margin: const EdgeInsets.only(top: 8.0),
            padding: const EdgeInsets.all(12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Expanded(
                  child: InkWell(
                    child: Column(
                      children: const [
                        Icon(
                          Icons.image,
                          size: 60.0,
                        ),
                        SizedBox(height: 12.0),
                        Text(
                          "Gallery",
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 16, color: Colors.black),
                        )
                      ],
                    ),
                    onTap: () {
                      _imgFromGallery();
                      Navigator.pop(context);
                    },
                  ),
                ),
                Expanded(
                  child: InkWell(
                    child: SizedBox(
                      child: Column(
                        children: const [
                          Icon(
                            Icons.camera_alt,
                            size: 60.0,
                          ),
                          SizedBox(height: 12.0),
                          Text(
                            "Camera",
                            textAlign: TextAlign.center,
                            style: TextStyle(fontSize: 16, color: Colors.black),
                          )
                        ],
                      ),
                    ),
                    onTap: () {
                      _imgFromCamera();
                      Navigator.pop(context);
                    },
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  _imgFromGallery() async {
    var image =
        await picker.pickImage(source: ImageSource.gallery, imageQuality: 50);
    if (image == null) return null;
    setState(() {
      _loading = true;
      imageFile = File(image.path);
    });

    // Resize the image to 224x224
    img.Image resizedImage = img.copyResize(
      img.decodeImage(File(image.path).readAsBytesSync())!,
      width: 224,
      height: 224,
    );

    await _runModelOnImage(resizedImage);
  }

  _imgFromCamera() async {
    var image =
        await picker.pickImage(source: ImageSource.camera, imageQuality: 50);
    if (image == null) return null;
    setState(() {
      _loading = true;
      imageFile = File(image.path);
    });

    // Resize the image to 224x224
    img.Image resizedImage = img.copyResize(
      img.decodeImage(File(image.path).readAsBytesSync())!,
      width: 224,
      height: 224,
    );

    await _runModelOnImage(resizedImage);
  }

  Future<void> _runModelOnImage(img.Image image) async {
    try {
      // Normalize the image
      Uint8List inputImageData = normalizeImage(image);

      // Check the input data size
      print('Input Data Size: ${inputImageData.lengthInBytes}');

      // Run inference using TensorFlow Lite
      var output = await Tflite.runModelOnBinary(
        binary: inputImageData,
        numResults: 6,
        threshold: 0.1,
      );

      // Check if the output is not null and not empty
      if (output != null && output.isNotEmpty) {
        // Access the first recognition result
        var firstResult = output[0];

        // Assuming the recognition result is a map with a 'label' key
        var label = firstResult['label'];

        // Do something with the label
        print('Recognition Label: $label');
      } else {
        print('No recognition results');
      }
    } catch (e) {
      print('Error in _runModelOnImage: $e');
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

 Uint8List normalizeImage(img.Image image) {
  var convertedBytes = Uint8List(270000); // Single channel, assuming grayscale
  var buffer = Uint8List.view(convertedBytes.buffer);
  int pixelIndex = 0;

  // Determine the normalization scale based on the model requirements
  double scale = 1.0 / 255.0;

  for (var i = 0; i < 224; i++) {
    for (var j = 0; j < 224; j++) {
      var pixel = image.getPixel(j, i);

      // Extract Red and Green components from the Pixel (assuming grayscale)
      var red = pixel.x;
      var green = pixel.y;

      // Convert to grayscale (you can adjust this conversion as needed)
      var grayscaleValue = 0.299 * red + 0.587 * green; // Adjust weights as needed

      // Normalize pixel values to the range [0, 1]
      buffer[pixelIndex++] = (grayscaleValue * scale * 255).toInt();
    }
  }

  return convertedBytes;
}

}
