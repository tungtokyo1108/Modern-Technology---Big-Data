/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/Acl.java
 *
 *  Created on: Jan 08, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.api.core.ApiFunction;
import com.google.api.services.bigquery.model.Dataset.Access;
import com.google.cloud.StringEnumType;
import com.google.cloud.StringEnumValue;
import java.io.Serializable;
import java.util.Objects;

/**
 * Access Control for BigQuery DataSet
 * Manage permissions on datasets. 
 * Dataset roles affect how you can access or modify the data inside of a project 
 */

 public final class Acl implements Serializable {

    private static final long serialVersionUID = 8357269726277191556L;
    private final Entity entity;
    private final Role role;

    /***
     * Datasets roles supported by BigQuery
     */
    public static final class Role extends StringEnumValue {
        private static final long serialVersionUID = -1992679397135956912L;

        private static final ApiFunction<String, Role> CONSTRUCTOR = 
            new ApiFunction<String, Role>() {
                @Override
                public Role apply(String constant) {
                    return new Role(constant);
                }
            };

        private static final StringEnumType<Role> type = new StringEnumType<>(Role.class, CONSTRUCTOR);

        /**
         * READER - read, query, copy or export tables in the dataset 
         * WRITER - READER + edit or append data in the data set 
         * OWNER  - WRITER + update and delete the dataset 
         */
        public static final Role READER = type.createAndRegister("READER");
        public static final Role WRITER = type.createAndRegister("WRITER");
        public static final Role OWNER = type.createAndRegister("OWNER");

        private Role(String constant)
        {
            super(constant);
        }
    }
 }
