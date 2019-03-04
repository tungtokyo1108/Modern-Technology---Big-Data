/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterCreateProperties.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterCreateProperties 
{
    @JsonProperty(value = "clusterVersion")
    private String clusterVersion;

    /**
     * Possible values include: "Windows", "Linux"
     */
    @JsonProperty(value = "osType")
    private OSType osType;

    /**
     * Possible values include: "Standard", "Premium"
     */
    @JsonProperty(value = "tier")
    private Tier tier;

    @JsonProperty(value = "clusterDefinition")
    private ClusterDefinition clusterDefinition;

    @JsonProperty(value = "securityProfile")
    private SecurityProfile securityProfile;

    @JsonProperty(value = "computeProfile")
    private ComputeProfile computeProfile;

    @JsonProperty(value = "storageProfile")
    private StorageProfile storageProfile;

    @JsonProperty(value = "diskEncrytionProperties")
    private DiskEncrytionProperties diskEncrytionProperties;

    public String clusterVersion()
    {
        return this.clusterVersion;
    }

    public ClusterCreateProperties clusterVersion(String clusterVersion)
    {
        this.clusterDefinition = clusterVersion;
        return this;
    }

    public OSType osType()
    {
        return this.osType;
    }

    public ClusterCreateProperties withOsType(OSType osType)
    {
        this.osType = osType;
        return this;
    }

    public Tier tier()
    {
        return this.tier;
    }

    public ClusterCreateProperties withTier(Tier tier) 
    {
        this.tier = tier;
        return this;
    }

    public ClusterDefinition clusterDefinition() {
        return this.clusterDefinition;
    }

    public ClusterCreateProperties withClusterDefinition(ClusterDefinition clusterDefinition)
    {
        this.clusterDefinition = clusterDefinition;
        return this;
    }

    public SecurityProfile securityProfile()
    {
        return this.securityProfile;
    }

    public ClusterCreateProperties withSecurityProfile(SecurityProfile securityProfile)
    {
        this.securityProfile = securityProfile;
        return this;
    }

    public ComputeProfile computeProfile() 
    {
        return this.computeProfile;
    }

    public ClusterCreateProperties withComputeProfile(ComputeProfile computeProfile)
    {
        this.computeProfile = computeProfile;
        return this;
    }

    public StorageProfile storageProfile()
    {
        return this.storageProfile;
    }

    public ClusterCreateProperties withStorageProfile(StorageProfile storageProfile)
    {
        this.storageProfile = storageProfile;
        return this;
    }

    public DiskEncrytionProperties diskEncrytionProperties() 
    {
        return this.diskEncrytionProperties;
    }

    public ClusterCreateProperties withDiskEncryptionProperties (DiskEncrytionProperties diskEncrytionProperties)
    {
        this.diskEncrytionProperties = diskEncrytionProperties;
        return this;
    }
}
