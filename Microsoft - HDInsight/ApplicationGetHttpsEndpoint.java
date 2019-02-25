/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/ApplicationGetHttpsEndpoint.java
 *
 *  Created on: Feb 25, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ApplicationGetHttpsEndpoint 
{
    @JsonProperty(value = "accessModes")
    private List<String> accessModes;

    @JsonProperty(value = "location")
    private String location;

    @JsonProperty(value = "destinationPort")
    private Integer destinationPort;

    @JsonProperty(value = "publicPort")
    private Integer publicPort;

    public List<String> accessModes()
    {
        return this.accessModes;
    }

    public ApplicationGetHttpsEndpoint withAccessModes(List<String> accessModes)
    {
        this.accessModes = accessModes;
        return this;
    }

    public String location()
    {
        return this.location;
    }

    public ApplicationGetHttpsEndpoint withLocation(String location)
    {
        this.location = location;
        return this;
    }

    public Integer destinationPort() 
    {
        return this.destinationPort;
    }

    public ApplicationGetHttpsEndpoint withDestinationPort(Integer destinationPort)
    {
        this.destinationPort = destinationPort;
        return this;
    }

    public Integer publicPort() 
    {
        return this.publicPort;
    }

    public ApplicationGetHttpsEndpoint withPublicPort(Integer publicPort)
    {
        this.publicPort = publicPort;
        return this;
    }
}
