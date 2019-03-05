/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterPatchParameters.java
 *
 *  Created on: Mar 05, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.Map;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterPatchParameters
{
    @JsonProperty(value = "tags")
    private Map<String, String> tags;

    public Map<String, String> tags()
    {
        return this.tags;
    }

    public ClusterPatchParameters withTags(Map<String, String> tags)
    {
        this.tags = tags;
        return this;
    }
}
