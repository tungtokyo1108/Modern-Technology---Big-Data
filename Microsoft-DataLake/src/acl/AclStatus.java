/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/acl/AclStatus.java
 *
 *  Created on: May 28, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.acl;

import java.util.List;

public class AclStatus
{
    public List<AclEntry> aclSpec;
    public String owner;
    public String group;
    public String octalPermissions;
    public boolean stickyBit;
}
