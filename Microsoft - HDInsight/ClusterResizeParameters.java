/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterResizeParameters.java
 *
 *  Created on: Mar 05, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterResizeParameters 
{
    @JsonProperty(value = "targetInstanceCount")
    private Integer targetInstanceCount;

    public Integer targetInstanceCount()
    {
        return this.targetInstanceCount;
    }

    public ClusterResizeParameters withTargetInstanceCount(Integer targetInstanceCount)
    {
        this.targetInstanceCount = targetInstanceCount;
        return this;
    }
}
