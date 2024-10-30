select  count(*) from postHistory as ph,  		posts as p,          users as u where ph.PostId = p.Id 	and p.OwnerUserId = u.Id  AND u.CreationDate>='2010-07-26 19:16:35'::timestamp;
