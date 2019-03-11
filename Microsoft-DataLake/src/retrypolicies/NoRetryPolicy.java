/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/retrypolicies/NoRetryPolicy.java
 *
 *  Created on: Mar 11, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.retrypolicies;

public class NoRetryPolicy implements RetryPolicy 
{
    private int retryCount = 0;
    private int waitInterval = 100;

    public boolean shouldRetry(int httpResponseCode, Exception lastException)
    {
        if (httpResponseCode == 401 && retryCount == 0)
        {
            wait(waitInterval);
            retryCount++;
            return true;
        }
        return false;
    }

    private void wait(int milliseconds)
    {
        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }
}
