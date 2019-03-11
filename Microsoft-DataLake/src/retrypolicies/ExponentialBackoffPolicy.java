/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/retrypolicies/ExponentialBackoffPolicy.java
 *
 *  Created on: Mar 11, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.retrypolicies;

/**
 * Implements different retry decisions based on the error
 */

public class ExponentialBackoffPolicy implements RetryPolicy
{
    private int retryCount = 0;
    private int maxRetries = 4;
    private int exponentialRetryInterval = 1000;
    private int exponentialFactor = 4;
    private long lastAttemptStartTime = System.nanoTime();

    public ExponentialBackoffPolicy() {}

    public ExponentialBackoffPolicy(int maxRetries, @Deprecated int linearRetryInterval, int exponentialRetryInterval)
    {
        this.maxRetries = maxRetries;
        this.exponentialRetryInterval = exponentialRetryInterval;
    }

    public ExponentialBackoffPolicy(int maxRetries, @Deprecated int linearRetryInterval, int exponentialRetryInterval, 
                                        int exponentialFactor)
    {
        this.maxRetries = maxRetries;
        this.exponentialRetryInterval = exponentialRetryInterval;
        this.exponentialFactor = exponentialFactor;
    }

    public boolean shouldRetry(int httpResponseCode, Exception lastException)
    {
        if ( (httpResponseCode >= 300 && httpResponseCode < 500
                                      && httpResponseCode != 408
                                      && httpResponseCode != 429
                                      && httpResponseCode != 401)
                || (httpResponseCode == 501)
                || (httpResponseCode == 505))
        {
            return false;
        }

        if (lastException != null || httpResponseCode >= 500 
                                  || httpResponseCode == 408
                                  || httpResponseCode == 429
                                  || httpResponseCode == 401)
        {
            if (retryCount < maxRetries)
            {
                int timeSpent = (int)((System.nanoTime() - lastAttemptStartTime) / 1000000);
                wait(exponentialRetryInterval - timeSpent);
                exponentialRetryInterval *= exponentialFactor;
                retryCount++;
                lastAttemptStartTime = System.nanoTime();
                return true;
            }
            else
            {
                return false;
            }
        }

        // There are not errors
        if (httpResponseCode >= 100 && httpResponseCode < 300)
        {
            return false;
        }

        return false;
    }

    private void wait(int milliseconds)
    {
        if (milliseconds <= 0)
        {
            return;
        }

        try {
            Thread.sleep(milliseconds);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }
}
