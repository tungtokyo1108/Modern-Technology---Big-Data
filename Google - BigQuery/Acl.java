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
import com.google.cloud.bigquery.BigQueryException;
import com.google.cloud.bigquery.TableId;
import com.google.rpc.RetryInfo;

import io.opencensus.stats.View;

import java.io.Serializable;
import java.util.Objects;

import javax.jws.soap.SOAPBinding.Use;

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

        public static Role valueOfStrict (String constant) 
        {
            return type.valueOfStrict(constant);
        }

        public static Role valueOf(String constant) 
        {
            return type.valueOf(constant);
        }

        public static Role[] values() 
        {
            return type.values();
        }
    }

    /**
     *  Grant access to the dataset
     */
    public abstract static class Entity implements Serializable
    {
        private static final long serialVersionUID = 8111776788607959944L;

        private final Type type;

        public enum Type
        {
            DOMAIN,
            GROUP,
            USER,
            VIEW
        }

        Entity(Type type)
        {
            this.type = type;
        }

        public Type getType() 
        {
            return type;
        }

        abstract Access toPb();

        static Entity fromPb(Access access)
        {
            if (access.getDomain() != null)
            {
                return new Domain(access.getDomain());
            }
            if (access.getGroupByEmail() != null)
            {
                return new Group(access.getGroupByEmail());
            }
            if (access.getSpecialGroup() != null)
            {
                return new Group(access.getSpecialGroup());
            }
            if (access.getUserByEmail() != null)
            {
                return new User(access.getUserByEmail());
            }
            if (access.getView() != null)
            {
                return new View(TableId.fromPb(access.getView()));
            }
            throw new BigQueryException(
                BigQueryException.UNKNOWN_CODE, "Unrecognized access configuration"
            );
        }
    }


    public static final class Domain extends Entity 
    {
        private static final long serialVersionUID = -3033025857280447253L;

        private final String domain;

        public Domain(String domain)
        {
            super(Type.DOMAIN);
            this.domain = domain;
        }

        public String getDomain() {
            return domain;
        }

        @Override
        public boolean equals(Object obj)
        {
            if (this == obj)
            {
                return true;
            }
            if (obj == null || getClass() != obj.getClass())
            {
                return false;
            }
            Domain domainEntity = (Domain) obj;
            return Objects.equals(getType(), domainEntity.getType()) 
                && Objects.equals(domain, domainEntity.getDomain());
        }

        @Override
        Access toPb() 
        {
            return new Access().setDomain(domain);
        }

        @Override
        public int hashCode() 
        {
            return Objects.hash(getType(),domain);
        }

        @Override
        public String toString()
        {
            return toPb().toString();
        }
    }

    public static final class Group extends Entity 
    {
        private static final String PROJECT_OWNERS = "projectOwners";
        private static final String PROJECT_READERS = "projectReaders";
        private static final String PROJECT_WRITERS = "projectWriters";
        private static final String ALL_AUTHENTICATED_USERS = "allAuthenticatedUsers";
        private static final long serialVersionUID = 5146829352398103029L;

        private final String identifier;

        public Group(String identifier) 
        {
            super(Type.GROUP);
            this.identifier = identifier;
        }

        public String getIdentifier()
        {
            return identifier;
        }

        @Override
        public boolean equals(Object obj)
        {
            if (this == obj)
            {
                return true;
            }
            if (obj == null || getClass() != obj.getClass())
            {
                return false;
            }
            Group group = (Group) obj;
            return Objects.equals(getType(), group.getType())
                && Objects.equals(identifier, group.identifier);
        }

        public int hashCode() 
        {
            return Objects.hash(getType(), identifier);
        }

        @Override
        public String toString()
        {
            return toPb().toString();
        }

        @Override
        Access toPb()
        {
            switch(identifier)
            {
                case PROJECT_OWNERS:
                    return new Access().setSpecialGroup(PROJECT_OWNERS);
                case PROJECT_READERS: 
                    return new Access().setSpecialGroup(PROJECT_READERS);
                case PROJECT_WRITERS:
                    return new Access().setSpecialGroup(PROJECT_WRITERS);
                case ALL_AUTHENTICATED_USERS:
                    return new Access().setSpecialGroup(ALL_AUTHENTICATED_USERS);
                default:
                    return new Access().setSpecialGroup(identifier);
            }
        }

        public static Group ofProjectOwners() 
        {
            return new Group(PROJECT_OWNERS);
        }

        public static Group ofProjectReaders() 
        {
            return new Group(PROJECT_READERS);
        }

        public static Group ofProjectWriters()
        {
            return new Group(PROJECT_WRITERS);
        }

        public static Group ofAllAuthenticatedUsers() 
        {
            return new Group(ALL_AUTHENTICATED_USERS);
        }
    }

    public static final class User extends Entity 
    {
        private static final long serialVersionUID = -4942821351073996141L;

        private final String email;

        public User(String email)
        {
            super(Type.USER);
            this.email = email;
        }

        public String getEmail()
        {
            return email;
        }

        @Override
        public boolean equals (Object obj) 
        {
            if (this == obj)
            {
                return true;
            }
            if (obj == null || getClass() != obj.getClass())
            {
                return false;
            }
            User user = (User) obj;
            return Objects.equals(getType(), user.getType())
                && Objects.equals(email, user.email);
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(getType(),email);
        }

        @Override
        public String toString()
        {
            return toPb().toString();
        }

        @Override
        Access toPb()
        {
            return new Access().setUserByEmail(email);
        }
    }

    
 }
