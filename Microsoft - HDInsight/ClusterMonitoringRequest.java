/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterMonitoringRequest.java
 *
 *  Created on: Mar 05, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterMonitoringRequest
{
    @JsonProperty(value = "workspaceId")
    private String workspaceId;

    @JsonProperty(value = "primaryKey")
    private String primaryKey;

    public String workspaceId()
    {
        return this.workspaceId;
    }

    public ClusterMonitoringRequest withWorkspaceId(String workspaceId)
    {
        this.withWorkspaceId = workspaceId;
        return this;
    }

    public String primaryKey()
    {
        return this.primaryKey;
    }

    public ClusterMonitoringRequest withPrimaryKey(String primaryKey)
    {
        this.primaryKey = primaryKey;
        return this;
    }
    
}
