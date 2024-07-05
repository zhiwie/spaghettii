package com.example.spaghetti

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import com.example.spaghetti.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var model: Model

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            binding = DataBindingUtil.setContentView(this, R.layout.activity_main)
            model = Model.newInstance(this)

            // Log the contents of the ml directory
            val files = assets.list("ml")
            Log.d("Files", "Files in ml directory: ${files?.joinToString()}")

            // Simple test case
            testModel()

            binding.buttonCompute.setOnClickListener {
                val urlText = binding.editUrl.text.toString()
                if (urlText.isNotEmpty()) {
                    try {
                        val result = runModelInference(urlText)
                        binding.testResult.text = "Harmfulness: ${result.first}% (${if (result.second) "Bad" else "Good"} link)"
                        binding.testResult.visibility = android.view.View.VISIBLE
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Error processing URL", e)
                        showToast("Error processing URL: ${e.message}")
                    }
                } else {
                    showToast("Please enter a URL")
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error initializing app", e)
            showToast("Error initializing app: ${e.message}")
        }
    }

    private fun runModelInference(urlText: String): Pair<Float, Boolean> {
        val inputArray = preprocessUrl(urlText)
        Log.d("runModelInference", "Input Array: ${inputArray.joinToString()}")

        val inputBuffer = ByteBuffer.allocateDirect(inputArray.size * 4).order(ByteOrder.nativeOrder())
        inputArray.forEach { inputBuffer.putFloat(it) }

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, inputArray.size), DataType.FLOAT32)
        inputFeature0.loadBuffer(inputBuffer)

        val outputs = model.process(inputFeature0)
        val outputArray = outputs.outputFeature0AsTensorBuffer.floatArray
        Log.d("runModelInference", "Raw Output Array: ${outputArray.joinToString()}")

        // Assuming the model outputs a single value between 0 and 1
        val outputValue = outputArray[0]
        val harmfulnessScore = outputValue * 100

        // Adjust this threshold based on your model's behavior
        val isHarmful = harmfulnessScore > 30

        Log.d("runModelInference", "Harmfulness Score: $harmfulnessScore, Is Harmful: $isHarmful")

        return Pair(harmfulnessScore, isHarmful)
    }

    private fun preprocessUrl(url: String): FloatArray {
        // Adjusting to provide 4000 bytes of input (1000 floats)
        val maxLength = 1000
        val charArray = url.toCharArray()
        val floatArray = FloatArray(maxLength)

        // Convert characters to floats
        for (i in charArray.indices) {
            if (i < maxLength) {
                floatArray[i] = charArray[i].toFloat() / 255f
            }
        }

        // Padding with zeros if the length is less than maxLength
        if (charArray.size < maxLength) {
            for (i in charArray.size until maxLength) {
                floatArray[i] = 0f
            }
        }

        return floatArray
    }

    private fun testModel() {
        val goodUrl = "https://www.example.com"
        val badUrl = "http://malicious-example.com/phishing"

        Log.d("TestModel", "Testing with good URL: $goodUrl")
        val goodResult = runModelInference(goodUrl)
        Log.d("TestModel", "Good URL result: $goodResult")

        Log.d("TestModel", "Testing with bad URL: $badUrl")
        val badResult = runModelInference(badUrl)
        Log.d("TestModel", "Bad URL result: $badResult")
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}
