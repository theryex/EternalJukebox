package org.abimon.eternalJukebox

import kotlinx.coroutines.CoroutineExceptionHandler
import org.slf4j.Logger
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

/**
 * Creates a named [CoroutineExceptionHandler] instance.
 * @param handler a function which handles exception thrown by a coroutine
 */
@Suppress("FunctionName")
public inline fun NamedCoroutineExceptionHandler(name: String, crossinline handler: (CoroutineContext, Throwable) -> Unit): CoroutineExceptionHandler =
    object : AbstractCoroutineContextElement(CoroutineExceptionHandler), CoroutineExceptionHandler {
        override fun handleException(context: CoroutineContext, exception: Throwable) =
            handler.invoke(context, exception)

        override fun toString(): String = name
    }

data class LogCoroutineExceptionHandler(val logger: Logger) : AbstractCoroutineContextElement(CoroutineExceptionHandler), CoroutineExceptionHandler {
    override fun handleException(context: CoroutineContext, exception: Throwable) {
        logger.error("[$context] An unhandled exception occurred", exception)
    }
}
