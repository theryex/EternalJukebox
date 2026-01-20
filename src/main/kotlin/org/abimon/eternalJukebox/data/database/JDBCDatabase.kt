package org.abimon.eternalJukebox.data.database

import com.zaxxer.hikari.HikariConfig
import com.zaxxer.hikari.HikariDataSource

object JDBCDatabase: HikariDatabase() {
    override val ds: HikariDataSource

    init {
        Class.forName("org.postgresql.Driver")
            .getDeclaredConstructor()
            .newInstance()

        val config = HikariConfig()
        config.jdbcUrl = databaseOptions["jdbcUrl"]?.toString() ?: throw IllegalStateException("jdbcUrl was not provided!")

        config.username = databaseOptions["username"]?.toString()
        config.password = databaseOptions["password"]?.toString()
        config.initializationFailTimeout = 0

        ds = HikariDataSource(config)

        initialise()
    }
}