/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/implementation/ClusterInner.java
 *
 *  Created on: Mar 05, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.microsoft.azure.management.hdinsight.v2018_06_01_preview.ClusterGetProperties;
import com.microsoft.azure.management.hdinsight.v2018_06_01_preview.ClusterIdentity;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.microsoft.rest.SkipParentValidation;
import com.microsoft.azure.Resource;

@SkipParentValidation
public class ClusterInner extends Resource 
{
    @JsonProperty(value = "etag")
    private String etag;

    @JsonProperty(value = "properties")
    private ClusterGetProperties properties;

    @JsonProperty(value = "identity")
    private ClusterIdentity identity;

    public String etag() 
    {
        return this.etag;
    }

    public ClusterInner withEtag(String etag)
    {
        this.etag = etag;
        return this;
    }

    public ClusterGetProperties properties()
    {
        return this.properties;
    }

    public ClusterInner withProperties(ClusterGetProperties properties)
    {
        this.properties = properties;
        return this;
    }

    public ClusterIdentity identity()
    {
        return this.identity;
    }

    public ClusterInner withidentity(ClusterIdentity identity) 
    {
        this.identity = identity;
        return this;
    }
}
