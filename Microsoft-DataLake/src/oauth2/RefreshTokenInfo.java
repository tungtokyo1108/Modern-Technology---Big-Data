/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/oauth2/RefreshTokenInfo.java
 *
 *  Created on: June 27, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.oauth2;

import java.util.Date;

public class RefreshTokenInfo {
    public String accessToken;
    public String refreshToken;
    public Date accessTokenExpiry;
}
