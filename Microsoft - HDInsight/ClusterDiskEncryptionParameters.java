/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ClusterDiskEncryptionParameters.java
 *
 *  Created on: Mar 04, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ClusterDiskEncryptionParameters 
{
    @JsonProperty(value = "vaultUri")
    private String vaultUri;

    @JsonProperty(value = "keyName")
    private String keyName;

    @JsonProperty(value = "keyVersion")
    private String keyVersion;

    public String vaultUri()
    {
        return this.vaultUri;
    }

    public ClusterDiskEncryptionParameters withVaultUri(String vaultUri)
    {
        this.vaultUri = vaultUri;
        return this;
    }

    public String keyVersion()
    {
        return this.keyVersion;
    }

    public ClusterDiskEncryptionParameters withKeyVersion(String keyVersion)
    {
        this.keyVersion = keyVersion;
        return this;
    }

    public String keyName()
    {
        return this.keyName;
    }

    public ClusterDiskEncryptionParameters withKeyName(String keyName)
    {
        this.keyName = keyName;
        return this;
    }
}
