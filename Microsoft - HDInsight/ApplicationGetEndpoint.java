/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ApplicationGetEndpoint.java
 *
 *  Created on: Feb 24, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ApplicationGetEndpoint 
{
    @JsonProperty(value = "location")
    private String location;

    @JsonProperty(value = "destinationPort")
    private Integer destinationPort;

    @JsonProperty(value = "publicPort")
    private Integer publicPort;

    public String location()
    {
        return this.location;
    }

    public ApplicationGetEndpoint withLocation(String location)
    {
        this.location = location;
        return this;
    }

    public Integer destinationPort()
    {
        return this.destinationPort;
    }

    public ApplicationGetEndpoint withDestinationPort(Integer destinationPort)
    {
        this.destinationPort = destinationPort;
        return this;
    }

    public Integer publicPort()
    {
        return this.publicPort;
    }

    public ApplicationGetEndpoint withPublicPort(Integer publicPort)
    {
        this.publicPort = publicPort;
        return this;
    }
}
