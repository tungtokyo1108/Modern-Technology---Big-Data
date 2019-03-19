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

    @Override
    public int read(byte[] b) throws IOException 
    {
        if (b == null)
        {
            throw new IllegalArgumentException("null byte array passed in to read() method");
        }
        return read(b, 0, b.length);
    }

    @Override 
    public int read(byte[] b, int off, int len) throws IOException
    {
        if (streamClosed) throw new IOException("attempting to read from a closed stream");
        if (b == null)
        {
            throw new IllegalArgumentException("null byte array passed in to read() method");
        }
        
        if (off < 0 || len < 0 || len > b.length - off) 
        {
            throw new IndexOutOfBoundsException();
        }
        if (log.isTraceEnabled())
        {
            log.trace("ADLFileInputStream.read(b,off,{}) at offset {} using client {} from file {}", len, getPos(), client.getClientId(), filename);
        }

        if (len == 0)
        {
            return 0;
        }

        if (bCursor == limit)
        {
            if (readFromService() < 0) return -1;
        }

        int bytesRemaining = limit - bCursor;
        int bytesToRead = Math.min(len, bytesRemaining);
        System.arraycopy(buffer, bCursor, b, off, bytesToRead);
        bCursor += bytesToRead;
        return bytesToRead;
    }

    /**
     * Read from service try to read bytes from service.
     * Return how many bytes are actually reads, could be less than blocksize
     */
    protected long readFromService() throws IOException
    {
        if (bCursor < limit) return 0;
        if (fCursor >= directoryEntry.length) return -1;

        if (directoryEntry.length <= blocksize)
        {
            return slurpFullFile();
        }

        bCursor = 0;
        limit = 0;
        if (buffer == null) buffer = new byte[blocksize];

        int bytesRead = readInternal(fCursor, buffer, 0, blocksize, false);
        limit += bytesRead;
        fCursor += bytesRead;
        return bytesRead;
    }

    /**
     * Reads the whole file into buffer
     */
    protected long slurpFullFile() throws IOException
    {
        if (log.isTraceEnabled())
        {
            log.trace("ADLFileInputStream.slurpFullFile() - using client {} from file {}. At offset {}", client.getClientId(), filename, getPos());
        }

        if (buffer == null)
        {
            blocksize = (int) directoryEntry.length;
            buffer = new byte[blocksize];
        }
        
        // Preserve current file offset
        bCursor = (int) getPos();
        limit = 0;
        // Read from begining
        fCursor = 0;
        int loopCount = 0;

        // If one Open request doesnt get full file, then read again at fCursor
        while (fCursor < directoryEntry.length)
        {
            int bytesRead = readInternal(fCursor, buffer, limit, blocksize - limit, true);
            limit += bytesRead;
            fCursor += bytesRead;

            // Defensive against infinite loops 
            loopCount++;
            if (loopCount >= 10)
            {
                throw new IOException("Too many attempts to read whole file" + filename);
            }
        }
        return fCursor;
    }

    public int read(long position, byte[] b, int offset, int length) throws IOException
    {
        if (streamClosed) throw new IOException("attempting to read from a closed stream");
        if (log.isTraceEnabled())
        {
            log.trace("ADLFileInputStream positioned read() - at offset {} using client {} from file {}", position, client.getClientId(), filename);
        }
        return readInternal(position, b, offset, length, true);
    }

    private int readInternal(long position, byte[] b, int offset, int length, boolean bypassReadAhead) throws IOException
    {
        boolean readAheadEnabled = true;
        if (readAheadEnabled && !bypassReadAhead && client.disableReadAheads)
        {
            // try reading from read-ahead
            if (offset != 0) throw new IllegalArgumentException("readahead buffers cannot have non-zero buffer offsets");
            int receivedBytes;

            // queue read-aheads
            int numReadAheads = this.readAheadQueueDepth;
            long nextSize;
            long nextOffset = position;
            while (numReadAheads > 0 && nextOffset < directoryEntry.length)
            {
                nextSize = Math.min((long)blocksize, directoryEntry.length - nextOffset);
                if (log.isTraceEnabled())
                {
                    log.trace("Queueing readAhead for file" + filename + "offset" + nextOffset + "thread" + Thread.currentThread().getName());
                }
                ReadBufferManager.getBufferManager().queueReadAhead(this, nextOffset, (int) nextSize);
                nextOffset = nextOffset + nextSize;
                numReadAheads--;   
            }

            // try reading from buffers first 
            receivedBytes = ReadBufferManager.getBufferManager().getBlock(this, position, length, b);
            if (receivedBytes > 0) return receivedBytes;
            
            // got nothing from read-ahead, do our own read now
            receivedBytes = readRemote(position, b, offset, length, false);
            return receivedBytes;
        }
        else 
        {
            return readRemote(position, b, offset, length, false);
        }
    }
}