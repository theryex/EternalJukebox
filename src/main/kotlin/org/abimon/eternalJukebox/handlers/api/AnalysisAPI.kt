package org.abimon.eternalJukebox.handlers.api

import com.github.kittinunf.fuel.Fuel
import com.github.kittinunf.fuel.coroutines.awaitStringResponseResult
import io.vertx.core.json.JsonArray
import io.vertx.core.json.JsonObject
import io.vertx.ext.web.Router
import io.vertx.ext.web.RoutingContext
import io.vertx.ext.web.handler.BodyHandler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.abimon.eternalJukebox.*
import org.abimon.eternalJukebox.objects.*
import org.abimon.visi.io.ByteArrayDataSource
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*

object AnalysisAPI : IAPI {
    override val mountPath: String = "/analysis"
    private val logger: Logger = LoggerFactory.getLogger("AnalysisApi")

    override fun setup(router: Router) {
        router.get("/analyse/:id").suspendingHandler(this::analyseSpotify)
        router.get("/search").suspendingHandler(AnalysisAPI::searchSpotify)
        router.get("/loaded").suspendingHandler(this::listLoadedSongs)
        router.post("/upload/:id")
            .handler(BodyHandler.create().setDeleteUploadedFilesOnEnd(true).setBodyLimit(10 * 1000 * 1000))
        router.post("/upload/:id").suspendingHandler(this::upload)
    }

    private suspend fun listLoadedSongs(context: RoutingContext) {
        val songs = mutableListOf<JsonObject>()
        
        // Get songs from ANALYSIS folder
        val analysisFolder = File(
            EternalJukebox.config.storageOptions["ANALYSIS_FOLDER"] as? String ?: "data/analysis"
        )
        if (analysisFolder.exists() && analysisFolder.isDirectory) {
            analysisFolder.listFiles { file -> file.extension == "json" }?.forEach { file ->
                try {
                    val content = withContext(Dispatchers.IO) { file.readText() }
                    val json = JsonObject(content)
                    val info = json.getJsonObject("info")
                    if (info != null) {
                        songs.add(jsonObjectOf(
                            "id" to (info.getString("id") ?: file.nameWithoutExtension),
                            "title" to (info.getString("title") ?: info.getString("name") ?: "Unknown"),
                            "artist" to (info.getString("artist") ?: "Unknown")
                        ))
                    }
                } catch (e: Exception) {
                    logger.debug("Failed to parse analysis file: ${file.name}", e)
                }
            }
        }
        
        // Get songs from UPLOADED_ANALYSIS folder
        val uploadedFolder = File(
            EternalJukebox.config.storageOptions["UPLOADED_ANALYSIS_FOLDER"] as? String ?: "data/uploaded_analysis"
        )
        if (uploadedFolder.exists() && uploadedFolder.isDirectory) {
            uploadedFolder.listFiles { file -> file.extension == "json" }?.forEach { file ->
                try {
                    val content = withContext(Dispatchers.IO) { file.readText() }
                    val json = JsonObject(content)
                    val info = json.getJsonObject("info")
                    if (info != null) {
                        val id = info.getString("id") ?: file.nameWithoutExtension
                        // Avoid duplicates
                        if (songs.none { it.getString("id") == id }) {
                            songs.add(jsonObjectOf(
                                "id" to id,
                                "title" to (info.getString("title") ?: info.getString("name") ?: "Unknown"),
                                "artist" to (info.getString("artist") ?: "Unknown")
                            ))
                        }
                    }
                } catch (e: Exception) {
                    logger.debug("Failed to parse uploaded analysis file: ${file.name}", e)
                }
            }
        }
        
        // Sort by title
        songs.sortBy { it.getString("title")?.lowercase() ?: "" }
        
        context.response()
            .putHeader("X-Client-UID", context.clientInfo.userUID)
            .putHeader("Content-Type", "application/json")
            .end(JsonArray(songs))
    }

    private suspend fun analyseSpotify(context: RoutingContext) {
        if (EternalJukebox.storage.shouldStore(EnumStorageType.ANALYSIS)) {
            val id = context.pathParam("id")
            if (EternalJukebox.storage.isStored("$id.json", EnumStorageType.ANALYSIS)) {
                if (EternalJukebox.storage.provide("$id.json", EnumStorageType.ANALYSIS, context, context.clientInfo))
                    return

                val data = EternalJukebox.storage.provide("$id.json", EnumStorageType.ANALYSIS, context.clientInfo)
                if (data != null)
                    return context.response().putHeader("X-Client-UID", context.clientInfo.userUID)
                        .end(data, "application/json")
            }

            if (EternalJukebox.storage.shouldStore(EnumStorageType.UPLOADED_ANALYSIS)) {
                if (EternalJukebox.storage.isStored("$id.json", EnumStorageType.UPLOADED_ANALYSIS)) {
                    if (EternalJukebox.storage.provide(
                            "$id.json",
                            EnumStorageType.UPLOADED_ANALYSIS,
                            context,
                            context.clientInfo
                        )
                    ) return

                    val data =
                        EternalJukebox.storage.provide(
                            "$id.json",
                            EnumStorageType.UPLOADED_ANALYSIS,
                            context.clientInfo
                        )
                    if (data != null)
                        return context.response().putHeader("X-Client-UID", context.clientInfo.userUID)
                            .end(data, "application/json")
                }

                // Let the frontend handle auto-analysis with progress bar
                return context.response().putHeader("X-Client-UID", context.clientInfo.userUID).setStatusCode(400).end(
                    jsonObjectOf(
                        "error" to "This track currently has no analysis data. Attempting to analyze...",
                        "show_manual_analysis_info" to true,
                        "client_uid" to context.clientInfo.userUID
                    )
                )
            }

            context.response().putHeader("X-Client-UID", context.clientInfo.userUID).setStatusCode(400).end(
                jsonObjectOf(
                    "error" to "It is not possible to get new analysis data from Spotify. Please check the subreddit linked under 'Social' in the navigation bar for more information.",
                    "client_uid" to context.clientInfo.userUID
                )
            )
        } else {
            context.response().putHeader("X-Client-UID", context.clientInfo.userUID).setStatusCode(501).end(
                jsonObjectOf(
                    "error" to "Configured storage method does not support storing ANALYSIS",
                    "client_uid" to context.clientInfo.userUID
                )
            )
        }
    }

    private suspend fun searchSpotify(context: RoutingContext) {
        val query = context.request().getParam("query") ?: "Never Gonna Give You Up"
        val results = EternalJukebox.spotify.search(query, context.clientInfo)

        context.response().end(JsonArray(results.map(JukeboxInfo::toJsonObject)))
    }

    private suspend fun upload(context: RoutingContext) {
        val id = context.pathParam("id")

        if (!EternalJukebox.storage.shouldStore(EnumStorageType.UPLOADED_ANALYSIS)) {
            return context.endWithStatusCode(502) {
                this["error"] = "This server does not support uploaded analysis"
            }
        } else if (context.fileUploads().isEmpty()) {
            return context.endWithStatusCode(400) {
                this["error"] = "No file uploads"
            }
        }

        val uploadedFile = File(context.fileUploads().first().uploadedFileName())
        var track: JukeboxTrack? = null

        val info = EternalJukebox.spotify.getInfo(id, context.clientInfo) ?: run {
            uploadedFile.guaranteeDelete()
            logger.warn("[{}] Failed to get track info for {}", context.clientInfo.userUID, id)
            return context.endWithStatusCode(400) {
                this["error"] = "Failed to get track info"
            }
        }

        try {
            val mapResponse = withContext(Dispatchers.IO) {
                EternalJukebox.jsonMapper.tryReadValue(uploadedFile.readBytes(), Map::class)
            } ?: return context.endWithStatusCode(400) { this["error"] = "Analysis file could not be parsed" }

            val obj = JsonObject(mapResponse.mapKeys { (key) -> "$key" })
            track = JukeboxTrack(
                info,
                withContext(Dispatchers.IO) {
                    JukeboxAnalysis(
                        EternalJukebox.jsonMapper.readValue(
                            obj.getJsonArray("sections").toString(),
                            Array<SpotifyAudioSection>::class.java
                        ),
                        EternalJukebox.jsonMapper.readValue(
                            obj.getJsonArray("bars").toString(),
                            Array<SpotifyAudioBar>::class.java
                        ),
                        EternalJukebox.jsonMapper.readValue(
                            obj.getJsonArray("beats").toString(),
                            Array<SpotifyAudioBeat>::class.java
                        ),
                        EternalJukebox.jsonMapper.readValue(
                            obj.getJsonArray("tatums").toString(),
                            Array<SpotifyAudioTatum>::class.java
                        ),
                        EternalJukebox.jsonMapper.readValue(
                            obj.getJsonArray("segments").toString(),
                            Array<SpotifyAudioSegment>::class.java
                        )
                    )
                },
                JukeboxSummary((mapResponse["track"] as Map<*, *>)["duration"].toString().toDouble())
            )

            if (track.info.duration != (track.audio_summary.duration * 1000).toInt()) {
                return context.endWithStatusCode(400) {
                    this["error"] = "Track duration does not match analysis duration. This is likely due to an incorrect analysis file. Make sure it is for the song ${info.name} by ${info.artist}"
                }
            }

            context.response().putHeader("X-Client-UID", context.clientInfo.userUID)
                .end(track.toJsonObject())

            withContext(Dispatchers.IO) {
                EternalJukebox.storage.store(
                    "$id.json",
                    EnumStorageType.UPLOADED_ANALYSIS,
                    ByteArrayDataSource(track.toJsonObject().toString().toByteArray(Charsets.UTF_8)),
                    "application/json",
                    context.clientInfo
                )
            }
        } finally {
            uploadedFile.guaranteeDelete()

            if (track == null) {
                context.endWithStatusCode(400) { this["error"] = "Analysis file could not be parsed" }
            } else {
                logger.info("[{}] Uploaded analysis for {}", context.clientInfo.userUID, id)
            }
        }
    }


    private suspend fun requestAnalysisFromFloppa(url: String, id: String, clientInfo: ClientInfo?): String? {
        val analyzerUrl = EternalJukebox.config.analyzerUrl ?: return null

        try {
            // 1. Submit analysis request
            val (_, response, _) = Fuel.post("$analyzerUrl/analyze/")
                .header("Content-Type", "application/json")
                .body(JsonObject().put("url", url).toString())
                .awaitStringResponseResult()

            if (response.statusCode != 200) {
                logger.warn("[{}] Failed to request analysis from Floppa: {}", clientInfo?.userUID, response.statusCode)
                return null
            }

            // 2. Poll for completion
            var attempt = 0
            while (attempt < 60) { // Timeout after ~3 minutes (60 * 3s)
                delay(3000)
                attempt++

                val (_, statusRes, _) = Fuel.get("$analyzerUrl/analyze/status/$id").awaitStringResponseResult()
                if (statusRes.statusCode != 200) continue

                val statusJson = JsonObject(String(statusRes.data, Charsets.UTF_8))
                val status = statusJson.getString("status")

                if (status == "completed") {
                    // Analysis is done and saved to disk.
                    // The Jukebox shares the volume, so we can now try to load it from UPLOADED_ANALYSIS storage type.
                    if (EternalJukebox.storage.isStored("$id.json", EnumStorageType.UPLOADED_ANALYSIS)) {
                        val dataSource =
                            EternalJukebox.storage.provide("$id.json", EnumStorageType.UPLOADED_ANALYSIS, clientInfo)
                        return dataSource?.let {
                            withContext(Dispatchers.IO) {
                                String(it.inputStream.readBytes(), Charsets.UTF_8)
                            }
                        }
                    }
                    // If not found yet, wait a bit more?
                } else if (status == "error") {
                    logger.warn("[{}] Floppa analysis failed: {}", clientInfo?.userUID, statusJson.getString("log"))
                    return null
                }
            }
        } catch (e: Exception) {
            logger.error("[{}] Error communicating with Floppa Analyzer", clientInfo?.userUID, e)
        }

        return null
    }

    init {
        logger.info("Initialised Analysis Api")
    }
}
