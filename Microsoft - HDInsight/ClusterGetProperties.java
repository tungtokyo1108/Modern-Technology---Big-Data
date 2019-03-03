/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterGetProperties.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterGetProperties 
{
    @JsonProperty(value = "clusterVersion")
    private String clusterVersion;

    @JsonProperty(value = "osType")
    private OSType osType;

    @JsonProperty(value = "tier")
    private Tier tier;

    @JsonProperty(value = "clusterDefinition", required = true)
    private ClusterDefinition clusterDefinition;

    @JsonProperty(value = "securityProfile")
    private SecurityProfile securityProfile;

    @JsonProperty(value = "computeProfile")
    private ComputeProfile computeProfile;

    @JsonProperty(value = "provisioningState")
    private HDInsightClusterProvisioningState provisioningState;

    @JsonProperty(value = "createdDate")
    private String createdDate;

    @JsonProperty(value = "clusterState")
    private String clusterState;

    @JsonProperty(value = "quotaInfo")
    private QuotaInfo quotaInfo;

    @JsonProperty(value = "errors")
    private List<Errors> errors;

    @JsonProperty(value = "connectivityEndpoints")
    private List<ConnectivityEndpoints> connectivityEndpoints;

    @JsonProperty(value = "diskEncryptionProperties")
    private DiskEncryptionProperties diskEncryptionProperties;

    public String clusterVersion() 
    {
        return this.clusterVersion;
    }

    public ClusterGetProperties withClusterVersion(String clusterVersion)
    {
        this.clusterVersion = clusterVersion;
        return this;
    }

    public OSType osType()
    {
        return this.osType;
    }

    public ClusterGetProperties withOsType(OSType osType)
    {
        this.osType = osType;
        return this;
    }

    public Tier tier()
    {
        return this.tier;
    }

    public ClusterGetProperties withTier(Tier tier)
    {
        this.tier = tier;
        return this;
    }

    public ClusterDefinition clusterDefinition()
    {
        return this.clusterDefinition;
    }

    public ClusterGetProperties withClusterDefinition(ClusterDefinition clusterDefinition)
    {
        this.clusterDefinition = clusterDefinition;
        return this;
    }

    public SecurityProfile securityProfile()
    {
        return this.securityProfile;
    }

    public ClusterGetProperties withSecurityProfile(SecurityProfile securityProfile)
    {
        this.securityProfile = securityProfile;
        return this;
    }
}
