/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterIdentity.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.Map;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Identity for the cluster 
 * - The principal id of cluster identity. This property will only be provided for a system assisgned identity
 * - The tenant id associated with the cluster. 
 * - The type of identity used for the cluster. "SystemAssigned, UserAssigned" includes both an implicitly 
 *   created identity and a set of user assigned identities. 
 * - The list of user identities associated with the cluster. The user identity dictionary key reference 
 *   will be ARM resources ids in the form
 */

public class ClusterIdentity 
{
    @JsonProperty(value = "principalId", access = JsonProperty.Access.WRITE_ONLY)
    private String principalId;

    @JsonProperty(value = "tenantId", access = JsonProperty.Access.WRITE_ONLY)
    private String tenantId;

    @JsonProperty(value = "type")
    private ResourceIdentityType type;

    @JsonProperty(value = "userAssignedIdentities")
    private Map<String, ClusterIdentityUserAssignedIdentitiesValue> userAssignedIdentities;

    public String principalId() 
    {
        return this.principalId;
    }

    public String tenantId() 
    {
        return this.tenantId;
    }

    public ResourceIdentityType type()
    {
        return this.type;
    }

    public ClusterIdentity withType(ResourceIdentityType type) 
    {
        this.type = type;
        return this;
    }

    public Map<String, ClusterIdentityUserAssignedIdentitiesValue> userAssignedIdentities()
    {
        return this.userAssignedIdentities;
    }

    public ClusterIdentity withUserAssignedIdentities(Map<String, ClusterIdentityUserAssignedIdentitiesValue> userAssignedIdentities)
    {
        this.userAssignedIdentities = userAssignedIdentities;
        return this;
    }
}
