package com.example.spaghetti

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import com.example.spaghetti.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.security.MessageDigest

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            binding = DataBindingUtil.setContentView(this, R.layout.activity_main)

            binding.buttonCompute.setOnClickListener {
                val urlText = binding.editUrl.text.toString()
                if (urlText.isNotEmpty()) {
                    try {
                        val harmfulnessPercentage = runModelInference(this, urlText)
                        binding.testResult.text = "Percentage of harmfulness: $harmfulnessPercentage%"
                        binding.testResult.visibility = android.view.View.VISIBLE
                    } catch (e: Exception) {
                        e.printStackTrace()
                        showToast("Error processing URL: ${e.message}")
                    }
                } else {
                    showToast("Please enter a URL")
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            showToast("Error initializing app: ${e.message}")
        }
    }

    private fun runModelInference(context: android.content.Context?, urlText: String): Float {
        return try {
            val inputArray = hashUrlToVector(urlText)
            val inputBuffer = ByteBuffer.allocateDirect(inputArray.size * 4).apply {
                inputArray.forEach { putFloat(it) }
            }

            context?.let { ctx ->
                val model = Model.newInstance(ctx)
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, inputArray.size), DataType.FLOAT32)
                inputFeature0.loadBuffer(inputBuffer)

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                model.close()

                outputFeature0.floatArray[0] * 100
            } ?: throw IllegalStateException("Context is null")
        } catch (e: Exception) {
            e.printStackTrace()
            throw e
        }
    }

    private fun hashUrlToVector(url: String, vectorSize: Int = 1000): FloatArray {
        val vector = FloatArray(vectorSize)
        val md = MessageDigest.getInstance("SHA-256")
        val hashBytes = md.digest(url.toByteArray())
        for (i in hashBytes.indices) {
            vector[i % vectorSize] += (hashBytes[i].toInt() and 0xFF) / 255.0f
        }
        return vector
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}