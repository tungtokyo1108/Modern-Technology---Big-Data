/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/ADLFileInputStream.java
 *
 *  Created on: Mar 15, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror;

import com.microsoft.azure.datalake.store.mirror.retrypolicies.ExponentialBackoffPolicy;
import com.microsoft.azure.datalake.store.mirror.retrypolicies.NoRetryPolicy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.util.UUID;

/**
 * ADLFFileInputStream can be used to read data from an open file on ADL.
 * It is a buffering stream, that reads data from the server in bulk, 
 * and then satisfies user reads from the buffer. 
 */

public class ADLFileInputStream extends InputStream
{
    private static final Logger log = LoggerFactory.getLogger("com.microsoft.azure.datalake.store.ADLFileInputStream");

    private final String filename;
    private final ADLStoreClient client;
    private final DirectoryEntry directoryEntry;
    private final String sessionId = UUID.randomUUID().toString();
    private static final int defaultQueueDepth = 0;

    // 4MB default buffer size
    private int blocksize = 4 * 1024 * 1024; 
    // Initialized on the first use
    private byte[] buffer = null;
    // Initialized in constructor 
    private int readAheadQueueDepth;

    // Cursor of buffer within file - offset of next byte to read from remote server 
    private long fCursor = 0;
    // Cursor of read within buffer - offset of next byte to be returned from buffer
    private int bCursor = 0;
    // offset of next byte to be read into buffer from services 
    private int limit = 0;
    private boolean streamClosed = false;

    ADLFileInputStream(String filename, DirectoryEntry de, ADLStoreClient client)
    {
        super();
        this.filename = filename;
        this.client = client;
        this.directoryEntry = de;
        int requestedQD = client.getReadAheadQueueDepth();
        this.readAheadQueueDepth = (requestedQD >= 0) ? requestedQD : defaultQueueDepth;
        if (log.isTraceEnabled())
        {
            log.trace("ADLFileInputStream created for client() for file {}", client.getClientId(), filename);
        }
    }

    @Override
    public int read() throws IOException
    {
        byte[] b = new byte[1];
        int i = read(b, 0, 1);
        if (i < 0) return i;
        else return (b[0] & 0xFF);
    }
}
