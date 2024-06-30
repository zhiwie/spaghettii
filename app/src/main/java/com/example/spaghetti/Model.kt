package com.example.spaghetti

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class Model(private val context: Context) {
    private var interpreter: Interpreter? = null

    init {
        val model = FileUtil.loadMappedFile(context, "model.tflite")
        interpreter = Interpreter(model)
    }

    fun process(inputFeature0: TensorBuffer): Outputs {
        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1), org.tensorflow.lite.DataType.FLOAT32)
        interpreter?.run(inputFeature0.buffer, outputFeature0.buffer)
        return Outputs(outputFeature0)
    }

    fun close() {
        interpreter?.close()
    }

    class Outputs(val outputFeature0AsTensorBuffer: TensorBuffer)

    companion object {
        fun newInstance(context: Context?): Model {
            return Model(context ?: throw IllegalArgumentException("Context cannot be null"))
        }
    }
}