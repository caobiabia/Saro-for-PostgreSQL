select  count(*) from comments as c,          postHistory as ph,  		badges as b,          users as u  where u.Id = c.UserId 	and u.Id = ph.UserId 	and u.Id = b.UserId  AND c.Score=0  AND c.CreationDate>='2010-10-02 18:27:20'::timestamp  AND ph.PostHistoryTypeId=5  AND u.UpVotes>=0  AND u.UpVotes<=86;