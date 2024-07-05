package com.example.spaghetti

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer

class Model(private val context: Context) {
    private var interpreter: Interpreter? = null

    init {
        try {
            val model = loadModelFile()
            val options = Interpreter.Options()
            interpreter = Interpreter(model, options)
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            Log.d("Model", "Input shape: ${inputShape?.joinToString()}")
            Log.d("Model", "Output shape: ${outputShape?.joinToString()}")
        } catch (e: Exception) {
            Log.e("Model", "Error initializing model", e)
            throw e
        }
    }


    private fun loadModelFile(): MappedByteBuffer {
        return FileUtil.loadMappedFile(context, "model.tflite")
    }

    fun process(inputFeature0: TensorBuffer): Outputs {
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf(1, 1)
        val outputFeature0 = TensorBuffer.createFixedSize(outputShape, org.tensorflow.lite.DataType.FLOAT32)
        try {
            Log.d("Model", "Running inference with input: ${inputFeature0.floatArray.joinToString()}")
            interpreter?.run(inputFeature0.buffer, outputFeature0.buffer)
            Log.d("Model", "Inference output: ${outputFeature0.floatArray.joinToString()}")
        } catch (e: Exception) {
            Log.e("Model", "Error running model inference", e)
            throw e
        }
        return Outputs(outputFeature0)
    }

    fun close() {
        interpreter?.close()
    }

    class Outputs(val outputFeature0AsTensorBuffer: TensorBuffer)

    companion object {
        fun newInstance(context: Context): Model {
            return Model(context)
        }
    }
}