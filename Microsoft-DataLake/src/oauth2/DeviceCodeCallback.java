/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/oauth2/DeviceCodeCallback.java
 *
 *  Created on: June 26, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.oauth2;

class DeviceCodeCallback {

    /**
     * Shows the login message device code for user. The default implementation shows on the console 
     * Subclasses can override the method to display message in a different way 
     */
    
    private static DeviceCodeCallback defaultInstance = new DeviceCodeCallback();

    /**
     * Show the message to the user, instructing them to log in using the brower 
     * This method displays the message on standard output 
     */
    public void showDeviceCodeMessage(DeviceCodeInfo dcInfo) {
        System.out.println(dcInfo.message);
    }

    public static DeviceCodeCallback getDefaultInstance() {
        return defaultInstance;
    }
}
