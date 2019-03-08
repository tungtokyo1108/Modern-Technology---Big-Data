/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/ADLException.java
 *
 *  Created on: Mar 08, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror;

import java.io.IOException;

public class ADLException extends IOException 
{
    public int httpResponseCode;
    public String httpResponseMessage;
    public String requestId = null;
    public int numRetries;
    public long lastCallLatency = 0;
    public long responseContentLength = 0;
    public String remoteExceptionName = null;
    public String remoteExceptionMessage = null;
    public String remoteExceptionJavaClassName = null;

    public ADLException(String message)
    {
        super(message);
    }

    public ADLException(String message, Throwable initCause)
    {
        super(message, initCause);
    }
}
