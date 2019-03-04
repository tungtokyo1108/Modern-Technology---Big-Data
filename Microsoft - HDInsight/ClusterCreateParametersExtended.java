/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterCreateParametersExtended.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.Map;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterCreateParametersExtended 
{
    @JsonProperty(value = "location")
    private String location;

    @JsonProperty(value = "tags")
    private Map<String, String> tags;

    @JsonProperty(value = "identity")
    private ClusterIdentity identity;

    @JsonProperty(value = "properties")
    private ClusterCreateParametersExtended properties;

    public String location() 
    {
        return this.location;
    }

    public ClusterCreateParametersExtended withLocation(String location)
    {
        this.location = location;
        return this;
    }

    public Map<String, String> tags()
    {
        return this.tags;
    }

    public ClusterCreateParametersExtended withTags(Map<String, String> tags)
    {
        this.tags = tags;
        return this;
    }

    public ClusterCreateProperties properties() 
    {
        return this.properties;
    }

    public ClusterCreateParametersExtended withProperties(ClusterCreateProperties properties)
    {
        this.properties = properties;
        return this;
    }

    public ClusterIdentity identity()
    {
        return this.identity;
    }

    public ClusterCreateParametersExtended withIdentity(ClusterIdentity identity)
    {
        this.identity = identity;
        return this;
    }
}
