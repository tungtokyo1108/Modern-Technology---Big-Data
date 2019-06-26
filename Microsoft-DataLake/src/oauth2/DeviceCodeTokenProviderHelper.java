/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/oauth2/DeviceCodeTokenProviderHelper.java
 *
 *  Created on: June 26, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.oauth2;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Date;

class DeviceCodeInfo {
    String usercode;
    String verificationUrl;
    String message;
    String devicecode;
    int pollingInterval;
    Date expiry;
    String clientId;
}
