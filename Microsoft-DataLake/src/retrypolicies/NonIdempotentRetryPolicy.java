/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/retrypolicies/NonIdempotentRetryPolicy.java
 *
 *  Created on: Mar 11, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.retrypolicies;

public class NonIdempotentRetryPolicy implements RetryPolicy
{
    private int retryCount401 = 0;
    private int waitInterval = 100;

    private int retryCount429 = 0;
    private int maxRetries = 4;
    private int exponentialRetryInterval = 1000;
    private int exponentialFactor = 4;

    public boolean shouldRetry(int httpResponseCode, Exception lastException)
    {
        if (httpResponseCode == 401 && retryCount401 == 0)
        {
            wait(waitInterval);
            retryCount401++;
            return true;
        }

        if (httpResponseCode == 429)
        {
            if (retryCount429 < maxRetries)
            {
                wait(exponentialRetryInterval);
                exponentialRetryInterval *= exponentialFactor;
                retryCount429++;
                return true;
            }
            else 
            {
                return false;
            }
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
