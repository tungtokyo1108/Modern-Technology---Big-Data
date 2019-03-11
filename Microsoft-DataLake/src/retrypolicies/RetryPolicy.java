/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/retrypolicies/RetryPolicy.java
 *
 *  Created on: Mar 11, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.retrypolicies;

public interface RetryPolicy 
{
    boolean shouldRetry(int httResponseCode, Exception lastException);
}
