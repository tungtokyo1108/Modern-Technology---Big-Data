/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterIdentityUserAssignedIdentitiesValue.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterIdentityUserAssignedIdentitiesValue 
{
    @JsonProperty(value = "principalId", access = JsonProperty.Access.WRITE_ONLY)
    private String principalId;

    @JsonProperty(value = "clientId", access = JsonProperty.Access.WRITE_ONLY)
    private String clientId;

    public String principalId()
    {
        return this.principalId;
    }

    public String clientId()
    {
        return this.clientId;
    }
}
